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

def Charbonnier_loss(y_pred, y_true):
    diff = y_pred - y_true
    return torch.mean(torch.sqrt(diff**2 + 1e-6))

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

def add_white_noise(data, snr_db: float):
    """
    向输入 3D 数据添加白噪声，满足指定信噪比 SNR（单位 dB）。

    Args:
        data (ndarray): 输入张量，形状 (n1, n2, n3)。
        snr_db (float): 期望信噪比，单位 dB。
        seed (int|None): 随机种子（可选）。

    Returns:
        ndarray: 添加白噪声后的数据，形状不变。
    """
    data = np.asarray(data, dtype=np.float32)
    flat = data.reshape(data.shape[0], -1)  # (n1, n2*n3)

    # 生成白噪声
    noise = np.random.randn(*flat.shape).astype(np.float32)

    # 根据信噪比缩放噪声
    signal_power = np.mean(flat ** 2)
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= math.sqrt(desired_noise_power / np.mean(noise ** 2))

    # 合成并恢复形状
    noisy = flat + noise
    return noisy.reshape(data.shape)

def add_random_erratic_noise(data,
                             snr_db: float,
                             n_spikes: int = 100,
                             spike_scale: float = 10.0
                             ):
    """
    给 3-D 数据添加带尖脉冲的白噪声（“随机错动噪声”）。

    Args:
        data (ndarray): 输入张量，形状 (n1, n2, n3)。
        snr_db (float): 期望信噪比，单位 dB。
        n_spikes (int): 随机脉冲的数量。
        spike_scale (float): 脉冲幅度放大倍数。
        seed (int|None): 随机种子，便于复现。

    Returns:
        ndarray: 加噪后的数据，形状与输入相同。
    """
    data = np.asarray(data, dtype=np.float32)
    flat = data.reshape(data.shape[0], -1)          # (n1, n2*n3)

    # --- 白噪声 ---
    noise = np.random.randn(*flat.shape).astype(np.float32)
    signal_power = np.mean(flat ** 2)
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= math.sqrt(desired_noise_power / np.mean(noise ** 2))

    # --- 注入尖脉冲（erratic spikes）---
    cols = flat.shape[1]
    spike_idx = np.random.choice(cols, size=n_spikes, replace=False)
    noise[:, spike_idx] *= spike_scale

    # --- 合成并恢复形状 ---
    noisy = flat + noise
    return noisy.reshape(data.shape)

def BP_LP_HP_filter_2D(din, dt, flo=0, fhi=None, nplo=6, nphi=6, phase=0, verb=0):
    """
    Bandpass filtering for 2D seismic data.

    Parameters
    ----------
    din : numpy.ndarray
        Input data (2D array)
    dt : float
        Sampling interval
    flo : float, optional
        Low frequency in band, default is 0
    fhi : float, optional
        High frequency in band, default is Nyquist
    nplo : int, optional
        Number of poles for low cutoff, default is 6
    nphi : int, optional
        Number of poles for high cutoff, default is 6
    phase : int, optional
        0 for minimum phase, 1 for zero phase, default is 0
    verb : int, optional
        Verbosity flag, default is 0

    Returns
    -------
    dout : numpy.ndarray
        Output data
    """
    if din.ndim != 2:
        raise ValueError("Input data must be a 2D array")

    n1, n2 = din.shape
    dout = np.zeros((n1, n2))

    fnyq = 0.5 / dt

    if fhi is None:
        fhi = fnyq

    if flo < 0 or fhi < 0:
        raise ValueError('Negative frequency not allowed')
    if flo > fhi:
        raise ValueError('flo must be less than fhi')
    if fhi > fnyq:
        raise ValueError('fhi must be less than Nyquist frequency (0.5/dt)')

    # 归一化到 [0,1] 区间
    flo /= fnyq
    fhi /= fnyq

    if nplo < 1:
        nplo = 1
    if nplo > 1 and not phase:
        nplo = nplo // 2
    if nphi < 1:
        nphi = 1
    if nphi > 1 and not phase:
        nphi = nphi // 2

    if verb:
        print(f"flo={flo} fhi={fhi} nplo={nplo} nphi={nphi}")

    if flo > 1e-4:
        blo_b, blo_a = butter(nplo, flo, btype='high')
    else:
        blo_b, blo_a = None, None

    if fhi < 0.5 - 1e-4:
        bhi_b, bhi_a = butter(nphi, fhi, btype='low')
    else:
        bhi_b, bhi_a = None, None

    for i in range(n2):
        trace = din[:, i]

        if blo_b is not None:
            if phase:
                trace = filtfilt(blo_b, blo_a, trace)
            else:
                trace = lfilter(blo_b, blo_a, trace)
                trace = trace[::-1]
                trace = lfilter(blo_b, blo_a, trace)
                trace = trace[::-1]

        if bhi_b is not None:
            if phase:
                trace = filtfilt(bhi_b, bhi_a, trace)
            else:
                trace = lfilter(bhi_b, bhi_a, trace)
                trace = trace[::-1]
                trace = lfilter(bhi_b, bhi_a, trace)
                trace = trace[::-1]

        dout[:, i] = trace

    return dout


def mf(D, nfw=7, ifb=1, axis=2):
    '''
    MF: median filter along first or second axis for 2D profile

    INPUT
    D:   	intput data
    nfw:    window size
    ifb:    if use padded boundary (if not, zero will be padded)
    axis:   along the vertical (1) or horizontal (2) axis

    OUTPUT
    D1:  	output data
    '''
    # nfw should be odd
    if np.mod(nfw, 2) == 0:
        nfw = nfw + 1

    if axis == 2:
        D = D.transpose()
    n1 = D.shape[0]
    n2 = D.shape[1]

    nfw2 = (nfw - 1) / 2
    nfw2 = int(nfw2)

    if ifb == 1:
        D = np.concatenate((np.flipud(D[0:nfw2, :]), D, np.flipud(D[n1 - nfw2:n1, :])), axis=0)
    else:
        D = np.concatenate((np.zeros([nfw2, n2]), D, np.zeros([nfw2, n2])), axis=0)
    # output data
    D1 = np.zeros([n1, n2])
    for i2 in range(0, n2):
        for i1 in range(0, n1):
            D1[i1, i2] = np.median(D[i1:i1 + nfw, i2])
    if axis == 2:
        D1 = D1.transpose()

    return D1


def svmf(D, nfw=7, ifb=1, axis=2, l1=2, l2=0, l3=2, l4=4):
    '''
    SVMF: space-varying median filter along first or second axis for 2D profile

    INPUT
    D:   	intput data
    nfw:    window size
    ifb:    if use padded boundary (if not, zero will be padded)
    axis:   along the vertical (1) or horizontal (2) axis

    OUTPUT
    D1:  	output data
    # 		 win_len: window length distribution
    '''
    n1 = D.shape[0]
    n2 = D.shape[1]

    Dtmp = mf(D, nfw, ifb, axis)
    medianv = np.sum(np.abs(Dtmp.flatten())) / (n1 * n2)

    # nfw should be odd
    if np.mod(nfw, 2) == 0:
        nfw = nfw + 1

    # calculate length
    win_len = np.zeros([n1, n2], dtype='int')
    for i2 in range(0, n2):
        for i1 in range(0, n1):
            if np.abs(Dtmp[i1, i2]) < medianv:
                if np.abs(Dtmp[i1, i2]) < medianv / 2:
                    win_len[i1, i2] = nfw + l1
                else:
                    win_len[i1, i2] = nfw + l2
            else:
                if np.abs(Dtmp[i1, i2]) > medianv * 2:
                    win_len[i1, i2] = nfw - l4
                else:
                    win_len[i1, i2] = nfw - l3
    if axis == 2:
        D = D.transpose()
        win_len = win_len.transpose()
    n1 = D.shape[0]
    n2 = D.shape[1]
    win_len2 = (win_len - 1) / 2
    win_len2 = win_len2.astype(int)

    nfw_b = (np.max([nfw + l1, nfw + l2]) - 1) / 2
    nfw_b = int(nfw_b)

    if ifb == 1:
        D = np.concatenate((np.flipud(D[0:nfw_b, :]), D, np.flipud(D[n1 - nfw_b:n1, :])), axis=0)
    else:
        D = np.concatenate((np.zeros([nfw_b, n2]), D, np.zeros([nfw_b, n2])), axis=0)

    # output data
    D1 = np.zeros([n1, n2])
    for i2 in range(0, n2):
        for i1 in range(0, n1):
            # 			print(nfw_b,win_len2[i1,i2],i1+nfw_b-win_len2[i1,i2],i1+nfw_b+win_len2[i1,i2])
            # 			print(D.shape)
            # 			print(i1,i2,n1,n2)
            D1[i1, i2] = np.median(D[i1 + nfw_b - win_len2[i1, i2]:i1 + nfw_b + win_len2[i1, i2] + 1, i2])
    win_len = win_len2 * 2 + 1

    if axis == 2:
        D1 = D1.transpose()
        win_len = win_len.transpose()

    return D1, win_len


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


def morlet(omega_x, omega_y, epsilon=1, sigma=1, omega_0=2):
    return np.exp(-sigma**2 * ((omega_x - omega_0)**2 + (epsilon * omega_y)**2) / 2)

def mexh(omega_x, omega_y, sigma_y=1, sigma_x=1, order=2):
    return -(2 * np.pi) * (omega_x**2 + omega_y**2)**(order / 2) * \
           np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)

def gaus(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
    return (1j * omega_x)**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)

def gaus_2(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
    return (1j * (omega_x + 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)

def gaus_3(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1, b=1, a=1):
    return (1j * (a * omega_x + b * 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)

def cauchy(omega_x, omega_y, cone_angle=np.pi / 6, sigma=1, l=4, m=4):
    dot1 = np.sin(cone_angle) * omega_x + np.cos(cone_angle) * omega_y
    dot2 = -np.sin(cone_angle) * omega_x + np.cos(cone_angle) * omega_y
    coef = (dot1 ** l) * (dot2 ** m)

    k0 = (l + m) ** 0.5 * (sigma - 1) / sigma
    rad2 = 0.5 * sigma * ((omega_x - k0)**2 + omega_y**2)
    pond = np.tan(cone_angle) * omega_x > abs(omega_y)
    wft = pond * coef * np.exp(-rad2)

    return wft

def dog(omega_x, omega_y, alpha=1.25):
    m = (omega_x**2 + omega_y**2) / 2
    wft = -np.exp(-m) + np.exp(-alpha**2 * m)

    return wft
wavelets = dict(
    morlet=morlet,
    mexh=mexh,
    gaus=gaus,
    gaus_2=gaus_2,
    gaus_3=gaus_3,
    cauchy=cauchy,
    dog=dog
)


class WaveletTransformException(Exception):
    pass
def _get_wavelet_mask(wavelet: str, omega_x: np.array, omega_y: np.array, **kwargs):
    assert omega_x.shape == omega_y.shape

    try:
        return wavelets[wavelet](omega_x, omega_y, **kwargs)

    except KeyError:
        raise WaveletTransformException('Unknown wavelet: {}'.format(wavelet))

@lru_cache(5)
def _create_frequency_plane(image_shape: tuple):
    assert len(image_shape) == 2

    h, w = image_shape
    w_2 = (w - 1) // 2
    h_2 = (h - 1) // 2

    w_pulse = 2 * np.pi / w * np.hstack((np.arange(0, w_2 + 1), np.arange(w_2 - w + 1, 0)))
    h_pulse = 2 * np.pi / h * np.hstack((np.arange(0, h_2 + 1), np.arange(h_2 - h + 1, 0)))

    xx, yy = np.meshgrid(w_pulse, h_pulse, indexing='xy')
    dxx_dyy = abs((xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0]))

    return xx, yy, dxx_dyy


def cwt_2d(x, scales, wavelet, **wavelet_args):
    assert isinstance(x, np.ndarray) and len(x.shape) == 2, 'x should be 2D numpy array'

    x_image = np.fft.fft2(x)
    xx, yy, dxx_dyy = _create_frequency_plane(x_image.shape)
    cwt = []
    wav_norm = []

    for scale_val in scales:
        mask = scale_val * _get_wavelet_mask(wavelet, scale_val * xx, scale_val * yy, **wavelet_args)
        cwt.append(np.fft.ifft2(x_image * mask))
        wav_norm.append((np.sum(abs(mask)**2)*dxx_dyy)**(0.5 / (2 * np.pi)))

    cwt = np.stack(cwt, axis=2)
    wav_norm = np.array(wav_norm)

    return cwt, wav_norm


def apply_agc_hilbert(data, window=50, eps=1e-6):
    from scipy.signal import hilbert
    from scipy.ndimage import uniform_filter1d
    """
    使用 Hilbert 包络进行 AGC 增益，适配数据维度为 [n_samples, n_traces]
    返回 agc_data, gain_record，shape 同输入
    """
    n_samples, n_traces = data.shape
    agc_data = np.zeros_like(data)
    gain_record = np.zeros_like(data)

    for i in range(n_traces):
        trace = data[:, i]
        envelope = np.abs(hilbert(trace))
        local_amp = uniform_filter1d(envelope, size=window, mode='nearest')
        gain = 1.0 / (local_amp + eps)
        agc_data[:, i] = trace * gain
        gain_record[:, i] = gain
    return agc_data, gain_record

def reverse_agc(agc_data, gain_record):
    return agc_data / (gain_record + 1e-6)


def das_picker_stalta(X, nsta=5, nlta=20, trigger_threshold=0.98):
    """
    First-arrival picking using STA/LTA algorithm.

    Parameters:
        X: Input signal (2D numpy array).
        nsta: Short time period.
        nlta: Long time period.

    Returns:
        O: On-set picker (1D numpy array).
        R: Reference signal (2D numpy array).
    """
    # Ensure X is a 2D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n1, n2 = X.shape
    O = np.zeros(n2)  # Onset picker output
    R = np.zeros((n1, n2))  # Reference output

    for i2 in range(n2):
        xn = X[:, i2]  # Get the i-th column of X

        e = das_stalta(xn, nsta, nlta)  # Calculate STA/LTA

        # ref = das_scale(e, 1)  # Scale (this may not be necessary)
        ref = e / np.max(np.abs(e))

        # Find the first index where ref exceeds 0.98
        n_onset = np.min(np.where(ref > trigger_threshold)[0]) if np.any(ref > trigger_threshold) else 0
        O[i2] = n_onset
        R[:, i2] = ref

    return O, R

def das_stalta(a, nsta, nlta):
    """
    Compute STA/LTA (Short Time Average / Long Time Average) ratio.

    Parameters:
    a (ndarray): Input signal.
    nsta (int): Short time period.
    nlta (int): Long time period.

    Returns:
    ndarray: STA/LTA ratio.
    """
    # Initialize STA and LTA with the input signal
    sta = a.copy()
    lta = a.copy()

    # Compute STA
    sta = np.cumsum(sta ** 2)
    sta[nsta:] = sta[nsta:] - sta[:-nsta]  # Subtract previous STA values
    sta = sta / nsta  # Average over nsta

    # Compute LTA
    lta = np.cumsum(lta ** 2)
    lta[nlta:] = lta[nlta:] - lta[:-nlta]  # Subtract previous LTA values
    lta = lta / nlta  # Average over nlta

    # Set initial LTA values to zero
    sta[:nlta - 1] = 0

    # Avoid division by zero
    idx = np.abs(lta) <= 1e-7
    lta[idx] += 1e-7

    # Compute STA/LTA ratio
    b = sta / lta

    return b