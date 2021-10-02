import torch
from torch import nn
from scipy import signal
from torch.nn import functional as torch_functional


class ModelCNN_Dilations(nn.Module):
    """1D convolutional neural network with dilations. Classifier of the gravitaitonal waves
    Inspired by the https://arxiv.org/pdf/1904.08693.pdf
    """

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv1d(3, 256, kernel_size=1), nn.ReLU())
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(256, 256, kernel_size=2, dilation=2 ** i),
                    nn.ReLU(),
                )
                for i in range(11)
            ]
        )
        self.out_conv = nn.Sequential(nn.Conv1d(256, 1, kernel_size=1), nn.ReLU())
        self.fc = nn.Linear(2049, 1)

    def forward(self, x):
        x = self.init_conv(x)
        for conv in self.convs:
            x = conv(x)
        x = self.out_conv(x)
        x = self.fc(x)
        x.squeeze_(1)
        return x


class Model1DCNN(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, initial_channnels=8):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, initial_channnels, kernel_size=64),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels, kernel_size=32),
            nn.MaxPool1d(kernel_size=8),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels * 2, kernel_size=32),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 2, kernel_size=16),
            nn.MaxPool1d(kernel_size=6),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 4, kernel_size=16),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(initial_channnels * 4, initial_channnels * 4, kernel_size=16),
            nn.MaxPool1d(kernel_size=4),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )
        fm_size = 11
        self.fc1 = nn.Sequential(
            nn.Linear(initial_channnels * 4 * fm_size, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        # print(x.shape)
        x = x.flatten(1)
        # x = x.mean(-1)
        # x = torch.cat([x.mean(-1), x.max(-1)[0]])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    
class Model1DCNNGEM(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, initial_channnels=8):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, initial_channnels, kernel_size=64),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels, kernel_size=32),
            GeM(kernel_size=8),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels * 2, kernel_size=32),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 2, kernel_size=16),
            GeM(kernel_size=6),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 4, kernel_size=16),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(initial_channnels * 4, initial_channnels * 4, kernel_size=16),
            GeM(kernel_size=4),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )
        fm_size = 11
        self.fc1 = nn.Sequential(
            nn.Linear(initial_channnels * 4 * fm_size, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        # print(x.shape)
        x = x.flatten(1)
        # x = x.mean(-1)
        # x = torch.cat([x.mean(-1), x.max(-1)[0]])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x    

#----------------------------- V0  ----------------------------------------------
class ExtractorMaxPool(nn.Sequential):
    def __init__(self, in_c=8, out_c=8, kernel_size=64, maxpool=8, act=nn.SiLU(inplace=True)):
        super().__init__(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_c), act,
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
            nn.MaxPool1d(kernel_size=maxpool),
        )

class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, act=nn.SiLU(inplace=True)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_planes), act,
            nn.Conv1d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_planes))
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_planes)
            )
        self.act = act
        
    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

class ModelIafoss(nn.Module):
    def __init__(self, n=8, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(ExtractorMaxPool(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),),
            nn.Sequential(ExtractorMaxPool(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),)
        ])
        self.conv = nn.Sequential(
            ResBlock(3*n,2*n,kernel_size=31,stride=4),
            ResBlock(2*n,2*n,kernel_size=31),
            ResBlock(2*n,4*n,kernel_size=15,stride=4),
            ResBlock(4*n,4*n,kernel_size=15),
            ResBlock(4*n,8*n,kernel_size=7,stride=4),
            ResBlock(8*n,8*n,kernel_size=7),
        )
        self.head = nn.Sequential(nn.Flatten(),
            nn.Linear(n*8*8,256),nn.BatchNorm1d(256),nn.Dropout(ps), act,
            nn.Linear(256, 256),nn.BatchNorm1d(256),nn.Dropout(ps), act,
            nn.Linear(256, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))

#----------------------------------------------V1    
    
class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

# using GeM
class Extractor(nn.Sequential):
    def __init__(self, in_c=8, out_c=8, kernel_size=64, maxpool=8, act=nn.SiLU(inplace=True)):
        super().__init__(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_c), act,
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
#             nn.MaxPool1d(kernel_size=maxpool),
            GeM(kernel_size=maxpool),
        )
    
class ModelIafossV1(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            ResBlock(3*n,3*n,kernel_size=31,stride=4), #512
            ResBlock(3*n,3*n,kernel_size=31), #128
            ResBlock(3*n,4*n,kernel_size=15,stride=4), #128
            ResBlock(4*n,4*n,kernel_size=15), #32
            ResBlock(4*n,8*n,kernel_size=7,stride=4), #32
            ResBlock(8*n,8*n,kernel_size=7), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))

#for SE-----------------------------------------------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, act=nn.SiLU(inplace=True),reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_planes), act,
            nn.Conv1d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_planes),
            SELayer(out_planes, reduction)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_planes)
            )
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

class ModelIafossV1SE(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31))
        ])
        self.conv = nn.Sequential(
            SEResBlock(3*n,3*n,kernel_size=31,stride=4), #512
            SEResBlock(3*n,3*n,kernel_size=31), #128
            SEResBlock(3*n,4*n,kernel_size=15,stride=4), #128
            SEResBlock(4*n,4*n,kernel_size=15), #32
            SEResBlock(4*n,8*n,kernel_size=7,stride=4), #32
            SEResBlock(8*n,8*n,kernel_size=7), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[1](x[:,1].unsqueeze(1)),
            self.ex[2](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))
    
#for CBAM-----------------------------------------------------------------------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, silu=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) #0.01,default momentum 0.1
        self.silu = nn.SiLU(inplace=True) if silu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.silu is not None:
            x = self.silu(x)
        return x
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 15
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, silu=True)#silu False
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale
    
class CBAMResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, act=nn.SiLU(inplace=True),reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_planes), act,
            nn.Conv1d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_planes),
            SELayer(out_planes, reduction),
            SpatialGate(),
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_planes)
            )
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))
    
class ModelIafossV1CBAM(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),CBAMResBlock(n,n,kernel_size=31,stride=4),
                          CBAMResBlock(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),CBAMResBlock(n,n,kernel_size=31,stride=4),
                          CBAMResBlock(n,n,kernel_size=31))
        ])
        self.conv = nn.Sequential(
            CBAMResBlock(3*n,3*n,kernel_size=31,stride=4), #512
            CBAMResBlock(3*n,3*n,kernel_size=31), #128
            CBAMResBlock(3*n,4*n,kernel_size=15,stride=4), #128
            CBAMResBlock(4*n,4*n,kernel_size=15), #32
            CBAMResBlock(4*n,8*n,kernel_size=7,stride=4), #32
            CBAMResBlock(8*n,8*n,kernel_size=7), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))    

#---------------------------------------------------------------------------------------------------  
    
    
class BasicBlockPool(nn.Module):
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
                nn.MaxPool1d(downsample,ceil_mode=True), # downsampling 
            )
            self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.MaxPool1d(downsample,ceil_mode=True),  # downsampling 
                )
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

class ModelIafossV1Pool(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            BasicBlockPool(3*n,3*n,kernel_size=31,downsample=4), #512
            BasicBlockPool(3*n,3*n,kernel_size=31), #128
            BasicBlockPool(3*n,4*n,kernel_size=15,downsample=4), #128
            BasicBlockPool(4*n,4*n,kernel_size=15), #32
            BasicBlockPool(4*n,8*n,kernel_size=7,downsample=4), #32
            BasicBlockPool(8*n,8*n,kernel_size=7), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))

#---------------------------------------------------------------------------------------------------  
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

class ModelIafossV1GeM(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlock(n,n,kernel_size=31,stride=4),
                          ResBlock(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=4), #512
            ResBlockGeM(3*n,3*n,kernel_size=31), #128
            ResBlockGeM(3*n,4*n,kernel_size=15,downsample=4), #128
            ResBlockGeM(4*n,4*n,kernel_size=15), #32
            ResBlockGeM(4*n,8*n,kernel_size=7,downsample=4), #32
            ResBlockGeM(8*n,8*n,kernel_size=7), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))

#-----------------------------------------------------------------------------

class ModelIafossV1GeMAll(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=4), #512
            ResBlockGeM(3*n,3*n,kernel_size=31), #128
            ResBlockGeM(3*n,4*n,kernel_size=15,downsample=4), #128
            ResBlockGeM(4*n,4*n,kernel_size=15), #32
            ResBlockGeM(4*n,8*n,kernel_size=7,downsample=4), #32
            ResBlockGeM(8*n,8*n,kernel_size=7), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))

#-----------------------------------------------------------------------------    
class AdaptiveConcatPool1dx3(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d`,`AdaptiveMaxPool1d` and 'GeM' "
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
        self.gemp = GeM(kernel_size=8)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x),self.gemp(x)], 1)
    
class ModelGeMx3(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=4), #512
            ResBlockGeM(3*n,3*n,kernel_size=31), #128
            ResBlockGeM(3*n,4*n,kernel_size=15,downsample=4), #128
            ResBlockGeM(4*n,4*n,kernel_size=15), #32
            ResBlockGeM(4*n,8*n,kernel_size=7,downsample=4), #32
            ResBlockGeM(8*n,8*n,kernel_size=7), #8
        )
        self.head = nn.Sequential(AdaptiveConcatPool1dx3(),nn.Flatten(),
            nn.Linear(n*8*3,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))

#-----------------------------------------------------------------------------
class ModelIafossV1GeMAllDeep(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=4), #512
            ResBlockGeM(3*n,3*n,kernel_size=31), #128
            ResBlockGeM(3*n,3*n,kernel_size=31), 
            ResBlockGeM(3*n,3*n,kernel_size=31), 
            ResBlockGeM(3*n,4*n,kernel_size=15,downsample=4), #128
            ResBlockGeM(4*n,4*n,kernel_size=15), #32
            ResBlockGeM(4*n,4*n,kernel_size=15),
            ResBlockGeM(4*n,4*n,kernel_size=15),
            ResBlockGeM(4*n,8*n,kernel_size=7,downsample=4), #32
            ResBlockGeM(8*n,8*n,kernel_size=7), #8
            ResBlockGeM(8*n,8*n,kernel_size=7),
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))
    
#---------------------------------------------------------------------------------------------------
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
    
 
class DeepStochastic(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=False), ps=0.5, stochastic_final_layer_proba=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
        ])
        proba_final_layer = stochastic_final_layer_proba
        num_block = 11
        self.proba_step = (1-proba_final_layer)/(num_block-1)
        self.survival_proba = [1-i*self.proba_step for i in range(num_block)]
        self.conv = nn.Sequential(
            StochasticDepthResBlockGeM(3*n,3*n,kernel_size=31,downsample=4,p=self.survival_proba[0]), #512
            StochasticDepthResBlockGeM(3*n,3*n,kernel_size=31,p=self.survival_proba[1]), #128
            StochasticDepthResBlockGeM(3*n,3*n,kernel_size=31,p=self.survival_proba[2]), 
            StochasticDepthResBlockGeM(3*n,3*n,kernel_size=31,p=self.survival_proba[3]), 
            StochasticDepthResBlockGeM(3*n,4*n,kernel_size=15,downsample=4,p=self.survival_proba[4]), #128
            StochasticDepthResBlockGeM(4*n,4*n,kernel_size=15,p=self.survival_proba[5]), #32
            StochasticDepthResBlockGeM(4*n,4*n,kernel_size=15,p=self.survival_proba[6]),
            StochasticDepthResBlockGeM(4*n,4*n,kernel_size=15,p=self.survival_proba[7]),
            StochasticDepthResBlockGeM(4*n,8*n,kernel_size=7,downsample=4,p=self.survival_proba[8]), #32
            StochasticDepthResBlockGeM(8*n,8*n,kernel_size=7,p=self.survival_proba[9]), #8
            StochasticDepthResBlockGeM(8*n,8*n,kernel_size=7,p=self.survival_proba[10]),
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))
    
#-----------------------------------------------------------------------------
class Deeper(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=4), #512
            ResBlockGeM(3*n,3*n,kernel_size=3), #128
            ResBlockGeM(3*n,3*n,kernel_size=3), 
            ResBlockGeM(3*n,3*n,kernel_size=3), 
            ResBlockGeM(3*n,4*n,kernel_size=15,downsample=4), #128
            ResBlockGeM(4*n,4*n,kernel_size=3), #32
            ResBlockGeM(4*n,4*n,kernel_size=3),
            ResBlockGeM(4*n,4*n,kernel_size=3),
            ResBlockGeM(4*n,8*n,kernel_size=7,downsample=4), #32
            ResBlockGeM(8*n,8*n,kernel_size=7), #8
            ResBlockGeM(8*n,8*n,kernel_size=7),
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))
    
class Deeper2(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList([
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
            nn.Sequential(Extractor(1,n,63,maxpool=2,act=act),ResBlockGeM(n,n,kernel_size=31,downsample=4),
                          ResBlockGeM(n,n,kernel_size=31)),
        ])
        self.conv = nn.Sequential(
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=2), #512
            ResBlockGeM(3*n,3*n,kernel_size=31), 
            ResBlockGeM(3*n,3*n,kernel_size=31), 
            ResBlockGeM(3*n,3*n,kernel_size=31,downsample=2), 
            ResBlockGeM(3*n,3*n,kernel_size=31), 
            ResBlockGeM(3*n,3*n,kernel_size=31), 
            ResBlockGeM(3*n,4*n,kernel_size=15,downsample=2), 
            ResBlockGeM(4*n,4*n,kernel_size=15), 
            ResBlockGeM(4*n,4*n,kernel_size=15), 
            ResBlockGeM(4*n,4*n,kernel_size=15,downsample=2),
            ResBlockGeM(4*n,4*n,kernel_size=15),
            ResBlockGeM(4*n,4*n,kernel_size=15), 
            ResBlockGeM(4*n,8*n,kernel_size=7,downsample=2),
            ResBlockGeM(8*n,8*n,kernel_size=7), 
            ResBlockGeM(8*n,8*n,kernel_size=7), 
            ResBlockGeM(8*n,8*n,kernel_size=7,downsample=2),
            ResBlockGeM(8*n,8*n,kernel_size=7),#8
            ResBlockGeM(8*n,8*n,kernel_size=7), 
        )
        self.head = nn.Sequential(AdaptiveConcatPool1d(),nn.Flatten(),
            nn.Linear(n*8*2,nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, nh),nn.BatchNorm1d(nh),nn.Dropout(ps), act,
            nn.Linear(nh, 1),
        )
    def forward(self, x):
        x = torch.cat([
            self.ex[0](x[:,0].unsqueeze(1)),
            self.ex[0](x[:,1].unsqueeze(1)),
            self.ex[1](x[:,2].unsqueeze(1))],1)
        return self.head(self.conv(x))

    

    
#-------------------------------------------------------------------V2    

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
            return torch_functional.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
#https://www.kaggle.com/iafoss/mish-activation
import torch.nn.functional as F
class MishFunction(torch.autograd.Function):
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
    
    
    
    
#-----------------------------------
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
#         self.proba_step = (1-proba_final_layer)/(num_block-1)
#         self.survival_proba = [1-i*self.proba_step for i in range(num_block)]
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
