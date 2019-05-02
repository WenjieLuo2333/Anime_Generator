# -*- coding: utf-8 -*-
"""
Implements models in https://arxiv.org/pdf/1708.05509.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def swish(x):
    return x * F.sigmoid(x)

class StableBCELoss(nn.modules.Module):
   def __init__(self):
         super(StableBCELoss, self).__init__()
   def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

        self.ac = nn.ReLU()

    def forward(self, x):
        y = self.ac(self.bn1(self.conv1(x)))
        # print y.size()
        return self.bn2(self.conv2(y)) + x


class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        #self.shuffler = nn.ConvTranspose2d(out_channels, in_channels, 4, 2, 1)
        self.bn = nn.BatchNorm2d(in_channels)

        self.ac = nn.ReLU()

    def forward(self, x):
        return self.ac(self.bn(self.shuffler(self.conv(x))))


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)

        self.ac = nn.LeakyReLU()

    def forward(self, x):
        y = self.ac(self.conv1(x))
        return self.ac(self.conv2(y) + x)

class DresidualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(DresidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

        self.ac = nn.ReLU()

    def forward(self, x):
        y = self.ac(self.bn1(self.conv1(x)))
        # print y.size()
        return self.bn2(self.conv2(y)) + x



class DBlock(nn.Module):
    def __init__(self, n=64, k=3, s=1):
        super(DBlock, self).__init__()

        self.block1 = ResBlock(n, k, n, s)
        self.block2 = ResBlock(n, k, n, s)

        self.conv1 = nn.Conv2d(n, 2*n, 4, stride=2, padding=1)

        self.ac = nn.LeakyReLU()

    def forward(self, x):
        x = self.block1(x)
        #x = self.block2(x)
        return self.ac(self.conv1(x))




# custom weights initialization called on G and D
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





class Generator(nn.Module):
    def __init__(self, n_residual_blocks=12, upsample_factor=6, tag_num=4):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.dense = nn.Linear(128+tag_num, 64*16*16)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        
        self.innerRes1 = residualBlock(in_channels = 64,n = 64);

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 64, 16, 16)
        x = self.relu(self.bn1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.relu(self.bn2(y)) + x
        k = 1
        for i in range(self.upsample_factor//2):
            # print x.size()
            x = self.__getattr__('upsample' + str(i+1))(x)
        #x = self.innerRes1(x)
        

        return self.tanh(self.conv3(x))



class ResBlock2(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(ResBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.conv4 = nn.Conv2d(n,2*n,k,stride=s, padding=1)

        self.conv3 = nn.Conv2d(in_channels,2*n,1,stride=1, padding=0)
        self.pool = nn.AvgPool2d(2)
        self.ac = nn.LeakyReLU()

    def forward(self, x):
        y = self.ac(self.conv1(x))
        y1 = self.pool(self.conv3(x))

        y2 = self.ac(self.conv2(y))
        y2 = self.pool(self.conv4(y2))
       

        return (y1+y2)




class Discriminator(nn.Module):
    def __init__(self, tag_num=4):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)

        #self.block1 = ResBlock2(in_channels=32,n=32)
        self.block2 = ResBlock2(in_channels=32,n=64)
        self.block3 = ResBlock2(in_channels=128,n=128)
        self.block4 = ResBlock2(in_channels=256,n=256)
        self.block5 = ResBlock2(in_channels=512,n=512)

        self.block6 = ResBlock2(in_channels=256,n=256)
        self.block7 = ResBlock2(in_channels=512,n=512)

        self.head1 = nn.Linear(1024*2*2, 1)
        self.head2 = nn.Linear(1024*2*2, tag_num)

        self.ac = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ac(self.conv1(x))

        #x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x1 = self.block4(x)
        x1 = self.block5(x1)
        #x1 = self.block51(x1)

        x2 = self.block6(x)
        x2 = self.block7(x2)
        #x2 = self.block71(x2)

        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        return self.head1(x1), self.head2(x2) 

class DBlock2(nn.Module):
    def __init__(self, n=64, k=3, s=1):
        super(DBlock2, self).__init__()

        self.block1 = ResBlock(n, k, n, s)
        self.block2 = ResBlock(n, k, n, s)

        self.conv1 = nn.Conv2d(n, n, 4, stride=2, padding=1)

        self.ac = nn.LeakyReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.ac(self.conv1(x))

class Discriminator2(nn.Module):
    def __init__(self, tag_num=4):
        super(Discriminator2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)

        self.block1 = DBlock(n=32)
        self.block2 = DBlock(n=64)
        self.block3 = DBlock(n=128)
        self.block4 = DBlock2(n=256)
        self.block5 = DBlock(n=256)
        self.block51 = DBlock(n=128)

        self.block6 = DBlock2(n=256)
        self.block7 = DBlock(n=256)
        self.block71 = DBlock(n=128)

        self.head1 = nn.Linear(512*4, 1)
        self.head2 = nn.Linear(512*4, tag_num)

        self.ac = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ac(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        #x1 = self.block51(x)
        x1 = self.block4(x)
        x1 = self.block5(x1)
        
        #x2 = self.block71(x)
        x2 = self.block6(x)
        x2 = self.block7(x2)
        

        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        return self.head1(x1), self.head2(x2) # Use with numerically stable torch.nn.BCEWithLogitsLoss() during training
        # return self.sigmoid(self.head1(x)), self.sigmoid(self.head2(x))

# g = Generator()
# d = Discriminator()
# testx = Variable(torch.randn(1,128+12))
# out = g(testx)
# print out.size()
# h1, h2 = d(out)
# print h1.size()
# print h2.size()

# x = Variable(torch.randn(1,1,128))
# y = Variable(torch.randn(1,1,18))
# z = torch.cat((x, y), 2)
# print z.size()
