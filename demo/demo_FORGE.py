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


path = "second_FORGE_DAS.mat"
data = h5py.File(path, 'r')
raw = data['d1']
raw = np.array(raw).T.astype('float32')
raw = raw[:, 60:]
print(raw.shape, raw.max(), raw.min())

n1, n2 = raw.shape
time_size = 24
trace_size = 24
time_shift = 4
trace_shift = 4
patch = cg_patch(raw, l1=time_size, l2=trace_size, o1=time_shift, o2=trace_shift)
print(patch.shape)

patch = torch.tensor(patch.T, dtype=torch.float32).unsqueeze(1)

input_patch = patch.clone().to(device)
print(input_patch.shape)

train_dataset = torch.utils.data.TensorDataset(input_patch, input_patch)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
testloader = torch.utils.data.DataLoader(train_dataset, batch_size=5000, shuffle=False)

net = NFADNet(in_C=time_size*trace_size, hidden_C=512).to(device)
optimizer = optim.Adam(net.parameters(), lr=5e-4)
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

        epoch_loss += train_loss.item() 

    avg_epoch_loss = epoch_loss / len(trainloader)  
    loop1.close() 
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

