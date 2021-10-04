import torch
from torch import nn
from scipy import signal
import torch.nn.functional as F

class GeM(nn.Module):
    '''
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    '''
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        with torch.cuda.amp.autocast(enabled=False):#to avoid NaN issue for fp16
            return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class Extractor(nn.Sequential):
    def __init__(self, in_c=8, out_c=8, kernel_size=64, maxpool=8, act=nn.SiLU(inplace=True)):
        super().__init__(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_c), act,
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
#             nn.MaxPool1d(kernel_size=maxpool),
            GeM(kernel_size=maxpool),
        )


class ResBlockGeM(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, downsample=1, act=nn.SiLU(inplace=True)):
        super().__init__()
        self.act = act
        if downsample != 1 or in_channels != out_channels:
            self.residual_function = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                GeM(kernel_size=downsample), # downsampling 
            )
            self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    GeM(kernel_size=downsample),  # downsampling 
                )#skip layers in residual_function, can try simple MaxPool1d
        else:
            self.residual_function = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))

class StochasticDepthResBlockGeM(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, downsample=1, act=nn.SiLU(inplace=False),p=1):
        super().__init__()
        self.p = p
        self.act = act

        if downsample != 1 or in_channels != out_channels:
            self.residual_function = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                GeM(kernel_size=downsample), # downsampling 
            )
            self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    GeM(kernel_size=downsample),  # downsampling 
                )#skip layers in residual_function, can try simple Pooling
        else:
            self.residual_function = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            self.shortcut = nn.Sequential()
            
    def survival(self):
        var = torch.bernoulli(torch.tensor(self.p).float())#,device=device)
        return torch.equal(var,torch.tensor(1).float().to(var.device,non_blocking=True))

    def forward(self, x):
        if self.training:#attribute inherited
            if self.survival():
                x = self.act(self.residual_function(x) + self.shortcut(x))
            else:
                x = self.act(self.shortcut(x))
        else:
            x = self.act(self.residual_function(x) * self.p + self.shortcut(x))  
        return x
    

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class MishFunction(torch.autograd.Function):
    #https://www.kaggle.com/iafoss/mish-activation
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)


def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)


class ModelIafossV2(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5, use_raw_wave=False):
        super().__init__()
        self.window = nn.Parameter(torch.FloatTensor(signal.windows.tukey(4096+2*2048,0.5)),requires_grad=False)
        self.avr_spec = nn.Parameter(torch.load("./input/avr_w0.pth"),requires_grad=False)
        self.use_raw_wave = use_raw_wave

        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,127,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4,act=act),
                          ResBlockGeM(n,n,kernel_size=31,act=act)),
            nn.Sequential(Extractor(1,n,127,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4,act=act),
                          ResBlockGeM(n,n,kernel_size=31,act=act))
        ])
        self.conv1 = nn.ModuleList([
            nn.Sequential(
            ResBlockGeM(1*n,1*n,kernel_size=31,downsample=4,act=act), #512
            ResBlockGeM(1*n,1*n,kernel_size=31,act=act)),
            nn.Sequential(
            ResBlockGeM(1*n,1*n,kernel_size=31,downsample=4,act=act), #512
            ResBlockGeM(1*n,1*n,kernel_size=31,act=act)),
            nn.Sequential(
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=4,act=act), #512
            ResBlockGeM(3*n,3*n,kernel_size=31,act=act)),#128
            ])
        self.conv2 = nn.Sequential(
            ResBlockGeM(6*n,4*n,kernel_size=15,downsample=4,act=act),
            ResBlockGeM(4*n,4*n,kernel_size=15,act=act),#128
            ResBlockGeM(4*n,8*n,kernel_size=7,downsample=4,act=act), #32
            ResBlockGeM(8*n,8*n,kernel_size=7,act=act), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )

    def forward(self, x):
        if self.use_raw_wave:
            #!!
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    shape = x.shape
                    c = x.view(shape[0]*shape[1],-1)
                    c = torch.cat([-c.flip(-1)[:,4096-2049:-1]+2*c[:,0].unsqueeze(-1),c,
                                      -c.flip(-1)[:,1:2049]+2*c[:,-1].unsqueeze(-1)],1)
                    avr_spec = self.avr_spec.repeat(shape[0],1).view(-1,self.avr_spec.shape[-1])
                    x = torch.fft.ifft(torch.fft.fft(c*self.window)/avr_spec).real
                    x = x.view(shape[0],shape[1],x.shape[-1])
                    x = x[:,:,2048:-2048]

        x0 = [self.ex[0](x[:,0].unsqueeze(1)),self.ex[0](x[:,1].unsqueeze(1)),
              self.ex[1](x[:,2].unsqueeze(1))]
        x1 = [self.conv1[0](x0[0]),self.conv1[0](x0[1]),self.conv1[1](x0[2]),
              self.conv1[2](torch.cat([x0[0],x0[1],x0[2]],1))]
        x2 = torch.cat(x1,1)
        return self.head(self.conv2(x2))
    
    
    
    
class V2StochasticDepth(nn.Module):#stocnot on ex
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=False), ps=0.5, proba_final_layer=.5, use_raw_wave=False):
        super().__init__()
        self.window = nn.Parameter(torch.FloatTensor(signal.windows.tukey(4096+2*2048,0.5)),requires_grad=False)
        self.avr_spec = nn.Parameter(torch.load("./input/avr_w0.pth"),requires_grad=False) 
        self.use_raw_wave = use_raw_wave
        
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,127,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4,act=act),
                          ResBlockGeM(n,n,kernel_size=31,act=act)),
            nn.Sequential(Extractor(1,n,127,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4,act=act),
                          ResBlockGeM(n,n,kernel_size=31,act=act))
        ])
        
        num_block = 10
        # self.proba_step = (1-proba_final_layer)/(num_block-1)
        # self.survival_proba = [1-i*self.proba_step for i in range(num_block)]
        self.proba_step = (1-proba_final_layer)/(num_block)
        self.survival_proba = [1-i*self.proba_step for i in range(1,num_block+1)]
        
        self.conv1 = nn.ModuleList([
            nn.Sequential(
            StochasticDepthResBlockGeM(1*n,1*n,kernel_size=31,downsample=4,act=act,p=self.survival_proba[0]), #512
            StochasticDepthResBlockGeM(1*n,1*n,kernel_size=31,act=act,p=self.survival_proba[1])),
            nn.Sequential(
            StochasticDepthResBlockGeM(1*n,1*n,kernel_size=31,downsample=4,act=act,p=self.survival_proba[2]), #512
            StochasticDepthResBlockGeM(1*n,1*n,kernel_size=31,act=act,p=self.survival_proba[3])),
            nn.Sequential(
            StochasticDepthResBlockGeM(3*n,3*n,kernel_size=31,downsample=4,act=act,p=self.survival_proba[4]), #512
            StochasticDepthResBlockGeM(3*n,3*n,kernel_size=31,act=act,p=self.survival_proba[5])),#128
            ])
        self.conv2 = nn.Sequential(
            StochasticDepthResBlockGeM(6*n,4*n,kernel_size=15,downsample=4,act=act,p=self.survival_proba[6]),
            StochasticDepthResBlockGeM(4*n,4*n,kernel_size=15,act=act,p=self.survival_proba[7]),#128
            StochasticDepthResBlockGeM(4*n,8*n,kernel_size=7,downsample=4,act=act,p=self.survival_proba[8]), #32
            StochasticDepthResBlockGeM(8*n,8*n,kernel_size=7,act=act,p=self.survival_proba[9]), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )

    def forward(self, x):
        if self.use_raw_wave:
            #!!
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    shape = x.shape
                    c = x.view(shape[0]*shape[1],-1)
                    c = torch.cat([-c.flip(-1)[:,4096-2049:-1]+2*c[:,0].unsqueeze(-1),c,
                                      -c.flip(-1)[:,1:2049]+2*c[:,-1].unsqueeze(-1)],1)
                    avr_spec = self.avr_spec.repeat(shape[0],1).view(-1,self.avr_spec.shape[-1])
                    x = torch.fft.ifft(torch.fft.fft(c*self.window)/avr_spec).real
                    x = x.view(shape[0],shape[1],x.shape[-1])
                    x = x[:,:,2048:-2048]
        x0 = [self.ex[0](x[:,0].unsqueeze(1)),self.ex[0](x[:,1].unsqueeze(1)),
              self.ex[1](x[:,2].unsqueeze(1))]
        x1 = [self.conv1[0](x0[0]),self.conv1[0](x0[1]),self.conv1[1](x0[2]),
              self.conv1[2](torch.cat([x0[0],x0[1],x0[2]],1))]
        x2 = torch.cat(x1,1)
        return self.head(self.conv2(x2))
