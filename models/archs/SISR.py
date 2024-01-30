from .common import Res_CA_List
import torch.nn as nn


class SISR_block(nn.Module):
    def __init__(self):
        super(SISR_block, self).__init__()
        

        self.head = nn.Conv2d(3,64,3,1,1)

        # define body module
        self.body1 = Res_CA_List(8,64)
        self.body2 = Res_CA_List(8,64)
        self.body3 = Res_CA_List(8,64)

        self.tail = nn.Conv2d(64,64,3,1,1)


    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        b1 = self.body1(x)
        b2 = self.body2(b1)
        b3 = self.body3(b2)

        res = self.tail(b3)

        res += x

        return res 
