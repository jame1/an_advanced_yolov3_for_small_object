import torch
from torch import nn
class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

class DCM(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(DCM,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)
        #print(hxin.shape)
        hx1 = self.rebnconv1(hxin)
        # print(hx1.shape)
        hx2 = self.rebnconv2(hx1)
        # print(hx2.shape)
        hx3 = self.rebnconv3(hx2)
        # print(hx3.shape)

        hx4 = self.rebnconv4(hx3)
        # print(hx4.shape)
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        # print(hx3d.shape)
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        # print(hx2d.shape)
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))
        # print(hx1d.shape)

        return hx1d + hxin