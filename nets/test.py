from nets.DCM import RSU4F
import torch

x=torch.randn(1,3,64,64)
net=RSU4F(in_ch=3,out_ch=24)
print(net(x).shape)