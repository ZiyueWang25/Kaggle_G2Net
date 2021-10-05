import timm
import torch
import torch.nn as nn
from nnAudio import Spectrogram
from scipy import signal
import torch.nn.functional as F


class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_angles(self, positions, indexes):
        d_model_tensor = torch.FloatTensor([[self.d_model]]).to(positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def forward(self, input_sequences):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        positions = torch.arange(input_sequences.size(1)).unsqueeze(1).to(input_sequences.device)  # [seq_len, 1]
        indexes = torch.arange(self.d_model).unsqueeze(0).to(input_sequences.device)  # [1, d_model]
        angles = self.get_angles(positions, indexes)  # [seq_len, d_model]
        angles[:, 0::2] = torch.sin(angles[:, 0::2])  # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2])  # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(input_sequences.size(0), 1, 1)  # [batch_size, seq_len, d_model]
        return position_encoding


class AttnBlock(nn.Module):
    def __init__(self, n=512, nheads=8, dim_feedforward=2048):
        super().__init__()
        self.pe = PositionalEncodingLayer(n)
        self.layers = nn.Sequential(nn.TransformerEncoderLayer(n, nheads, dim_feedforward),
                                    nn.TransformerEncoderLayer(n, nheads, dim_feedforward))

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            shape = x.shape
            x = x.view(shape[0], shape[1], -1).permute(2, 0, 1).float()
            x = x + self.pe(x)
            x = self.layers(x)
            x = x.permute(1, 2, 0).reshape(shape)
        return x


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Model_Iafoss(nn.Module):
    def __init__(self, n=1, arch='resnext50_32x4d_ssl',
                 path='facebookresearch/semi-supervised-ImageNet1K-models', ps=0.5, use_raw_wave=False):
        super().__init__()
        self.Q_TRANSFORM2 = Spectrogram.CQT1992v2(sr=2048, fmin=22, n_bins=64, hop_length=32, output_format='Magnitude',
                                                  norm=1, bins_per_octave=12, window='nuttall')
        self.window = nn.Parameter(torch.FloatTensor(signal.windows.tukey(4096 + 2 * 2048, 0.5)), requires_grad=False)
        self.avr_spec = nn.Parameter(torch.load("/home/input/avr_w0.pth"), requires_grad=False)
        self.use_raw_wave = use_raw_wave

        m = torch.hub.load(path, arch)
        nc = list(m.children())[-1].in_features
        self.enc = nn.Sequential(*(list(m.children())[:3]) + list(m.children())[4:-2])

        nh = 768
        self.head = nn.Sequential(nn.Conv2d(nc, nh, (8, 1)),
                                  AttnBlock(nh),
                                  AdaptiveConcatPool2d(1),
                                  nn.Flatten(),
                                  nn.BatchNorm1d(2 * nh),
                                  nn.Dropout(ps / 2),
                                  nn.Linear(2 * nh, 512),
                                  nn.ReLU(inplace=True),
                                  nn.BatchNorm1d(512),
                                  nn.Dropout(ps),
                                  nn.Linear(512, n)
                                  )

    def forward(self, x):
        if self.use_raw_wave:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    shape = x.shape
                    c = x.view(shape[0] * shape[1], -1)
                    c = torch.cat([-c.flip(-1)[:, 4096 - 2049:-1] + 2 * c[:, 0].unsqueeze(-1), c,
                                   -c.flip(-1)[:, 1:2049] + 2 * c[:, -1].unsqueeze(-1)], 1)
                    avr_spec = self.avr_spec.repeat(shape[0], 1).view(-1, self.avr_spec.shape[-1])
                    c = torch.fft.ifft(torch.fft.fft(c * self.window) / avr_spec).real

                    t = self.Q_TRANSFORM2(c)
                    t = t.view(shape[0], shape[1], t.shape[1], t.shape[2])
                    t = t[:, :, :, 64 + 64 - 8:192 - 8]
                    t = (8.0 * t + 1.0).log()
                    t = F.interpolate(t, size=(128, 128), mode='bilinear', align_corners=True)

        return self.head(self.enc(t))

    ##### 1D + 2D models ######


def batchnorm_1d(nf: int):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm1d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(1.)
    return bn


class LinearBlock(nn.Module):
    def __init__(self, in_sz, out_sz, activation=nn.ReLU(inplace=True), ps=None):
        super().__init__()
        if ps is not None:
            self.block = nn.Sequential(nn.Linear(in_sz, out_sz), activation, batchnorm_1d(out_sz), nn.Dropout(ps))
        else:
            self.block = nn.Sequential(nn.Linear(in_sz, out_sz), activation, batchnorm_1d(out_sz))
        nn.init.kaiming_normal_(self.block)

    def forward(self, x):
        return self.block(x)


class Combined1D2D(nn.Module):
    def __init__(self, model_1d, model_2d, emb_1d=128, emb_2d=128, first=256, ps=0):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d

        self.window = nn.Parameter(torch.FloatTensor(signal.windows.tukey(4096 + 2 * 2048, 0.5)), requires_grad=False)
        self.avr_spec = nn.Parameter(torch.load("/home/input/avr_w0.pth"), requires_grad=False)
        self.spec_transform = Spectrogram.CQT1992v2(sr=2048, fmin=15, n_bins=64, hop_length=32,
                                                    output_format='Magnitude', norm=1, bins_per_octave=12,
                                                    window='nuttall')

        # Replace last linear layer to return a embedding of size emb_1d
        head = list(self.model_1d.head.children())
        new_linear = nn.Linear(head[-1].in_features, emb_1d)
        self.model_1d.head = nn.Sequential(*head[:-1] + [new_linear])

        # Replace last linear layer to return a embedding of size emb_2d
        old_head = self.model_2d.encoder.fc
        self.model_2d.encoder.fc = nn.Linear(old_head.in_features, emb_2d)

        self.head = nn.Sequential(
            nn.Linear(emb_1d + emb_2d, 512), nn.BatchNorm1d(512), nn.Dropout(0.25), nn.SiLU(inplace=True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.Dropout(0.5), nn.SiLU(inplace=True),
            nn.Linear(256, 1),
        )

    def freeze_conv(self, req_grad):
        for name, param in self.model_2d.named_parameters():
            if 'encoder.fc.' in name:
                continue
            param.requires_grad = req_grad

        for name, param in self.model_1d.named_parameters():
            if 'head.9.' in name:
                continue
            param.requires_grad = req_grad

    def frequency_encoding(self, x):
        # for 2D model
        device = x.device
        bs, fbins, t = x.shape[0], x.shape[2], x.shape[3]
        freq_encoding = 2 * torch.arange(fbins, device=device) / fbins - 1  # -1 to +1
        freq_encoding = torch.stack([freq_encoding] * t, -1).unsqueeze(0)
        freq_encoding = torch.stack([freq_encoding] * bs)
        return torch.cat([x, freq_encoding], 1)

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                shape = x.shape
                c = x.view(shape[0] * shape[1], -1)
                c = torch.cat([-c.flip(-1)[:, 4096 - 2049:-1] + 2 * c[:, 0].unsqueeze(-1), c,
                               -c.flip(-1)[:, 1:2049] + 2 * c[:, -1].unsqueeze(-1)], 1)
                avr_spec = self.avr_spec.repeat(shape[0], 1).view(-1, self.avr_spec.shape[-1])
                x = torch.fft.ifft(torch.fft.fft(c * self.window) / avr_spec).real
                x_1d = x.view(shape[0], shape[1], x.shape[-1])[:, :, 2048:-2048]

                x_2d = self.spec_transform(x)
                x_2d = x_2d.reshape(shape[0], shape[1], x_2d.shape[1], x_2d.shape[2])
                x_2d = x_2d[:, :, :, 64 + 64 - 8:192 - 8]
                x_2d = (8.0 * x_2d + 1.0).log()
                x_2d = F.interpolate(x_2d, size=(256, 256), mode='bilinear', align_corners=True)
                # spec = standard_scaler(spec)
                x_2d = self.frequency_encoding(x_2d)

        out_1d = self.model_1d(x_1d)
        out_2d = self.model_2d(x_2d)
        embedding = torch.cat([out_1d, out_2d], -1)
        return self.head(embedding)
