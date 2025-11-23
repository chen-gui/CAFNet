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

clean = np.load('syn_DAS/syn_DAS-clean.npy')
print(clean.shape, clean.max(), clean.min())

noisy = np.load('syn_DAS/syn_DAS-noisy.npy')
print(cg_snr(clean, noisy))

bp = BP_LP_HP_filter_2D(noisy, 0.002, flo=0, fhi=70, nplo=6, nphi=6, phase=0)
bpmf = mf(bp,2,1,2)

n1, n2 = noisy.shape
time_size = 16
trace_size = 16
time_shift = 1
trace_shift = 1

patch = cg_patch(noisy, l1=time_size, l2=trace_size, o1=time_shift, o2=trace_shift)
print(patch.shape)

patch = torch.tensor(patch.T, dtype=torch.float32).unsqueeze(1)
input_patch = patch.clone().to(device)
print(input_patch.shape)

train_dataset = torch.utils.data.TensorDataset(input_patch, input_patch)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
testloader = torch.utils.data.DataLoader(train_dataset, batch_size=5000, shuffle=False)

net = NFADNet(in_C=time_size*trace_size, hidden_C=512).to(device)
optimizer = optim.Adam(net.parameters(), lr=5e-4)
loss_fun = nn.MSELoss()

show_interval = 10
num_epoch = 10
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
        # train_loss = loss_fun(output, tranlabel) + loss_fun(output_weakdenoiser, tranlabel)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        loop1.set_postfix(loss=train_loss.item())

        epoch_loss += train_loss.item()  # 累加每个 batch 的 loss

    avg_epoch_loss = epoch_loss / len(trainloader)  # 计算平均损失
    loop1.close()  # <- 手动关闭进度条
    tqdm.write(f"[Epoch {epoch + 1}] Average Training Loss: {avg_epoch_loss:.6f}")
    total_loss.append(avg_epoch_loss)

    if (epoch + 1) % show_interval == 0:
        net.eval()
        out_data = torch.zeros((1, 1, time_size*trace_size), dtype=torch.float32)
        with torch.no_grad():
            loop2 = tqdm(enumerate(testloader), total=len(testloader), desc="Predicting")
            for batch_idx, (testindex, tranlabel) in loop2:
                testindex = testindex.to(device)
                output, output_weakdenoiser = net(testindex)
                out_data = torch.cat([out_data, output_weakdenoiser.detach().cpu()], 0)

        denoised_data = cg_patch_inv(out_data[1:].squeeze().numpy().T, n1, n2, l1=time_size, l2=trace_size, o1=time_shift, o2=trace_shift)
        print(cg_snr(clean, denoised_data))
        # np.save('Discussion/syn_DAS-pro-{}-16patch1step-using70%traningData.npy'.format(int(epoch + 1)), denoised_data)

        x = np.arange(1, clean.shape[1], 1)
        t = np.arange(0, 0.002*clean.shape[0], 0.002)
        clip = 2
        kwargs_imshow = dict(vmin=-clip, vmax=clip, interpolation=None, aspect='auto', cmap=cseis(),
                             extent=(x.min(), x.max(), t.max(), t.min()))
        _, ax = plt.subplots(1, 7, figsize=(18, 5), sharex=True, sharey=True)
        ax[0].imshow(clean, **kwargs_imshow)
        ax[0].set_title("raw")

        ax[1].imshow(bp, **kwargs_imshow)
        ax[1].set_title("bp")

        ax[2].imshow(bpmf, **kwargs_imshow)
        ax[2].set_title("bpmf")

        ax[3].imshow(denoised_data, **kwargs_imshow)
        ax[3].set_title("our")

        ax[4].imshow(noisy-denoised_data, **kwargs_imshow)
        ax[4].set_title("our Removed noise")

        ax[5].imshow(noisy, **kwargs_imshow)
        ax[5].set_title("noisy")

        ax[6].imshow(clean-denoised_data, **kwargs_imshow)
        ax[6].set_title("our Removed noise")
        plt.tight_layout()
        plt.show()
# np.save('Discussion/LossCurve-pro-syn_DAS.npy', np.array(total_loss, dtype=np.float32))
