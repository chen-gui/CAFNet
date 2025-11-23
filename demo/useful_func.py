import torch
import torch.nn as nn
import numpy as np
import math
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
import math
from scipy.signal import butter, filtfilt, lfilter
from functools import lru_cache
from efficient_kan import KAN

def cg_snr(clean, recon):
    snr = 20 * np.log10(np.linalg.norm(clean) / np.linalg.norm(clean - recon))
    return np.round(snr, 2)

def logcosh_loss_softplus_approx(y_pred, y_true):
    diff = y_pred - y_true
    return torch.mean(F.softplus(2.0 * diff) - diff - math.log(2.0))

# -------- 通道注意力模块（适配输入维度 [B, 1, C]） --------
class SEBlock_FC_Softmax(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)  # 对通道维度 softmax
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(channels * 2, channels)
        self.kan = KAN([channels, channels], base_activation=nn.SiLU, grid_range=[-1, 1])
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):  # x: [B, 1, C]
        w_logits = self.avgpool(x.transpose(1, 2)).transpose(1, 2)  # [B, 1, C]
        w = self.softmax(w_logits)         # [B, 1, C]
        out = x * w                       # [B, 1, C]
        # 残差拼接
        out = self.out(torch.cat([x, out], dim=-1))
        out = self.kan(out)
        # out = self.act(out)

        return out

class compactblock(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.kan1 = KAN([out_dim, out_dim], base_activation=nn.SiLU, grid_range=[-1, 1])
        # 新增多头自注意力模块
        self.mhsa = nn.MultiheadAttention(embed_dim=out_dim, num_heads=n_heads, batch_first=True)

        self.linear2 = nn.Linear(out_dim + in_dim, out_dim)
        self.kan2 = KAN([out_dim, out_dim], base_activation=nn.SiLU, grid_range=[-1, 1])

        self.se = SEBlock_FC_Softmax(out_dim)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.kan1(x1)
        # x1 = self.act(x1)

        x1_attn, _ = self.mhsa(x1, x1, x1)

        x2 = self.linear2(torch.cat([x1_attn, x], dim=-1))
        x2 = self.kan2(x2)
        # x2 = self.act(x2)

        x3 = self.se(x2)

        if self.use_residual:
            out = x3 + x1
        else:
            out = x2
        return out  # [B, 1, out_dim]

class weaker_denoiser(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.kan1 = KAN([out_dim, out_dim], base_activation=nn.SiLU, grid_range=[-1, 1])
        # 新增多头自注意力模块
        self.mhsa = nn.MultiheadAttention(embed_dim=out_dim, num_heads=n_heads, batch_first=True)

        self.linear2 = nn.Linear(out_dim, out_dim)
        self.kan2 = KAN([out_dim, out_dim], base_activation=nn.SiLU, grid_range=[-1, 1])

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.kan1(x1)
        # x1 = self.act(x1)
        x_attn, _ = self.mhsa(x1, x1, x1)
        x2 = self.linear2(x_attn)
        x2 = self.kan2(x2)
        # x2 = self.act(x2)
        return x2

class NFADNet(nn.Module):
    def __init__(self, in_C=64, hidden_C=512):
        super().__init__()
        self.layer1 = compactblock(in_C, hidden_C)
        self.layer2 = compactblock(hidden_C, hidden_C // 2)
        self.layer3 = compactblock(hidden_C // 2, hidden_C//4)
        self.layer4 = compactblock(hidden_C // 4, hidden_C)
        self.layer5 = compactblock(hidden_C, in_C)
        self.weakdenoiser = weaker_denoiser(in_C, in_C)
    def forward(self, x):  # x: [B, 1, patch_dim]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x_weak = self.weakdenoiser(x)
        return x, x_weak


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]

def cseis():
    seis = np.concatenate(
        (np.concatenate((0.5 * np.ones([1, 40]), np.expand_dims(np.linspace(0.5, 1, 88), axis=1).transpose(),
                             np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                            axis=1).transpose(),
            np.concatenate((0.25 * np.ones([1, 40]), np.expand_dims(np.linspace(0.25, 1, 88), axis=1).transpose(),
                             np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                            axis=1).transpose(),
            np.concatenate((np.zeros([1, 40]), np.expand_dims(np.linspace(0, 1, 88), axis=1).transpose(),
                             np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                            axis=1).transpose()), axis=1)
    return ListedColormap(seis)


def fkdip(d, w):
    '''
    fkdip: FK dip filter

    INPUT
    d: 	input data (2D)
    w:  half width (in percentage) of the cone filter (i.e., w*nk=nwidth)

    OUTPUT
    d0: output data

    EXAMPLE
    import pyseistr as ps
    from pyseistr import gensyn
    from pyseistr import genflat
    import numpy as np
    data=gensyn();[nt,nx]=data.shape;
    noise=genflat(nt,nx,t=np.linspace(5,10*36,37,dtype='int32'),amp=0.5*np.ones(37),freq=80);
    datan=data+noise;
    datafk=datan-ps.fkdip(datan,0.005); ## FK filtering
    import matplotlib.pyplot as plt;
    plt.figure(figsize=(8, 8));
    plt.subplot(1,3,1);
    plt.imshow(datan,aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');
    plt.subplot(1,3,2);
    plt.imshow(datafk,aspect='auto');plt.xlabel('Trace');
    plt.subplot(1,3,3);
    plt.imshow(datan-datafk,aspect='auto');plt.xlabel('Trace');
    plt.show();

    '''

    [n1, n2] = d.shape
    nf = nextpow2(n1)
    nk = nextpow2(n2)
    nf2 = int(nf / 2)
    nk2 = int(nk / 2)
    Dfft1 = np.fft.fft(d, nf, 0)
    Dtmp = Dfft1[0:nf2 + 1, :]

    # processing area
    Dtmp2 = np.fft.fft(Dtmp, nk, 1)
    Dtmp2 = np.fft.fftshift(Dtmp2, 1)

    nw = w * nk
    [nn1, nn2] = Dtmp2.shape
    mask = np.zeros([nn1, nn2])

    for i1 in range(1, nn1 + 1):
        for i2 in range(1, nn2 + 1):
            if i1 > (nn1 / nw) * (i2 - (nk2)) and i1 > (nn1 / nw) * ((nk2) - i2):
                mask[i1 - 1, i2 - 1] = 1

    Dtmp2 = Dtmp2 * mask
    Dtmp = np.fft.ifft(np.fft.ifftshift(Dtmp2, 1), nk, 1)

    # honor symmetry for inverse fft
    Dfft2 = np.zeros([nf, nk], dtype=np.complex_)
    Dfft2[0:nf2 + 1, :] = Dtmp
    Dfft2[nf2 + 1:, :] = np.conj(np.flipud(Dtmp[1:-1, :]))
    d0 = np.real(np.fft.ifft(Dfft2, nf, 0))
    d0 = d0[0:n1, 0:n2]

    return d0


def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n
