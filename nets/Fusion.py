import torch
import torch.nn as nn
import torch.nn.functional as F
class fusion(nn.Module):

    def __init__(self):
        super(fusion, self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.avgpool=nn.AvgPool2d(kernel_size=3,stride=2,padding=1)

        self.conv1=nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv1_1=nn.Conv2d(768,256,1)
    def forward(self,left,middle,right):
        x1=self.maxpool(left)
        x1=F.normalize(x1)
        x1=self.conv1(x1)
        x1=F.relu(x1)
        x2=self.avgpool(middle)
        x2=F.normalize(x2)
        x2=self.conv2(x2)
        x2 = F.relu(x2)
        sum=torch.cat([x1,x2,right],dim=1)
        sum=self.conv1_1(sum)

        return sum
if __name__ == '__main__':
    left=torch.randn(1,64,80,80)
    middle = torch.randn(1, 128, 40, 40)
    right=torch.randn(1,256,20,20)
    fusion=fusion()
    fusion(left,middle,right)




