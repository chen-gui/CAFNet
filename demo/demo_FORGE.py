import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import segyio
# print(segyio.__file__)
import pylops
import sys
print(sys.executable)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from useful_func import *
import h5py
from CG_patch_unpatch2D_3D import cg_patch, cg_patch_inv
from collections import defaultdict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''3, 38, 69, 75'''
i = 69
path = "F:\FORGE_DAS_data/mat_raw/eq-{}.mat".format(i)
data = h5py.File(path, 'r')
raw = data['d1']
raw = np.array(raw).T.astype('float32')
raw = raw[:, 60:]
print(raw.shape, raw.max(), raw.min())

bp = BP_LP_HP_filter_2D(raw, 0.0005, flo=0, fhi=200, nplo=8, nphi=8, phase=0)
bpmf = mf(bp,5,1,2)
# fk = bp-fkdip(bp, 0.03)

n1, n2 = raw.shape
time_size = 24
trace_size = 24
time_shift = 4
trace_shift = 4
patch_bpmf = cg_patch(raw, l1=time_size, l2=trace_size, o1=time_shift, o2=trace_shift)
print(patch_bpmf.shape)

patch_bpmf = torch.tensor(patch_bpmf.T, dtype=torch.float32).unsqueeze(1)

input_patch = patch_bpmf.clone().to(device)
print(input_patch.shape)

train_dataset = torch.utils.data.TensorDataset(input_patch, input_patch)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
testloader = torch.utils.data.DataLoader(train_dataset, batch_size=5000, shuffle=False)

net = NFADNet(in_C=time_size*trace_size, hidden_C=512).to(device)
optimizer = optim.Adam(net.parameters(), lr=5e-4)
lossfun = nn.MSELoss()
num_epoch = 10
show_interval = 10
total_loss = []
for epoch in range(num_epoch):
    net.train()
    loop1 = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Train-Epoch [{epoch+1}/{num_epoch}]")
    epoch_loss = 0.0
    for batch_idx, (tranindex, tranlabel) in loop1:
        tranindex = tranindex.to(device)
        tranlabel = tranlabel.to(device)

        output, output_weakdenoiser = net(tranindex)
        train_loss = logcosh_loss_softplus_approx(output, tranlabel) + logcosh_loss_softplus_approx(output_weakdenoiser, tranlabel)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        loop1.set_postfix(loss=train_loss.item())

        epoch_loss += train_loss.item()  # 累加每个 batch 的 loss

    avg_epoch_loss = epoch_loss / len(trainloader)  # 计算平均损失
    loop1.close()  # <- 手动关闭进度条
    tqdm.write(f"[Epoch {epoch + 1}] Average Training Loss: {avg_epoch_loss:.6f}")
    total_loss.append(avg_epoch_loss)

    if (epoch + 1) % show_interval== 0:
        net.eval()
        out_data = torch.zeros((1, 1, time_size*trace_size), dtype=torch.float32)
        with torch.no_grad():
            loop2 = tqdm(enumerate(testloader), total=len(testloader), desc="Predicting")
            for batch_idx, (testindex, tranlabel) in loop2:
                testindex = testindex.to(device)
                output, output_weakdenoiser = net(testindex)
                out_data = torch.cat([out_data, output_weakdenoiser.detach().cpu()], 0)

        denoised_data = cg_patch_inv(out_data[1:].squeeze().numpy().T, n1, n2, l1=time_size, l2=trace_size, o1=time_shift, o2=trace_shift)

        fk = denoised_data-fkdip(denoised_data, 0.04)

        # np.save('FORGE/FORGE-69-pro-{}-20251110.npy'.format(int(epoch + 1)), fk)


        x = np.arange(60, 960, 1)
        t = np.arange(0, 0.0005 * raw.shape[0], 0.0005)
        clip = 20
        kwargs_imshow = dict(vmin=-clip, vmax=clip, interpolation=None, aspect='auto', cmap=cseis(),
                             extent=(x.min(), x.max(), t.max(), t.min()))
        _, ax = plt.subplots(1, 7, figsize=(18, 5), sharex=True, sharey=True)
        ax[0].imshow(raw, **kwargs_imshow)
        ax[0].set_title("raw")

        ax[1].imshow(bp, **kwargs_imshow)
        ax[1].set_title("bp")

        ax[2].imshow(bpmf, **kwargs_imshow)
        ax[2].set_title("bpmf")

        ax[3].imshow(denoised_data, **kwargs_imshow)
        ax[3].set_title("our")

        ax[4].imshow(raw-denoised_data, **kwargs_imshow)
        ax[4].set_title("our Removed noise")

        ax[5].imshow(fk, **kwargs_imshow)
        ax[5].set_title("FK")

        ax[6].imshow(raw-fk, **kwargs_imshow)
        ax[6].set_title("our Removed noise")
        plt.tight_layout()
        plt.show()
np.save('Discussion/LossCurve-pro-second_FORGE_DAS.npy', np.array(total_loss, dtype=np.float32))
