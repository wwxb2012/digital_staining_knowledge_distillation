from tkinter.tix import Form
from cv2 import FlannBasedMatcher
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import argparse
from lib.nn import SynchronizedBatchNorm2d as SynBN2d
import functools
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomAffine
import random
eps=1e-7
# from ..trainer.utils import ToTensorhdr

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9,ndf=64):
        super(Generator, self).__init__()

        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ndf, 7),
                      nn.InstanceNorm2d(ndf),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = ndf
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(ndf, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc,ndf=32):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(ndf*2),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(ndf*4),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf*4, ndf*8, 4, padding=1),
                  nn.InstanceNorm2d(ndf*8),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(ndf*8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'
    return input, pad_left, pad_right, pad_top, pad_bottom
def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



class SIGGRAPHGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=functools.partial(nn.InstanceNorm2d,affine=False), use_tanh=True, classification=False):
        super(SIGGRAPHGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model1+=[norm_layer(64),]
        model1+=[nn.ReLU(True),]
        # model1+=[nn.ReflectionPad2d(1),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation

        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model2+=[norm_layer(128),]
        model2+=[nn.ReLU(True),]
        # model2+=[nn.ReflectionPad2d(1),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # Conv7
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        # model47+=[norm_layer(256),]
        model8=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model8+=[norm_layer(256),]
        model8+=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above        

        # model9=[norm_layer(128),]
        model9=[nn.ReLU(True),]
        # model9+=[nn.ReflectionPad2d(1),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10=[nn.ReLU(True),]
        # model10+=[nn.ReflectionPad2d(1),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        # classification output
        model_class=[nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]

        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        if(use_tanh):
            model_out+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'),])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, x):
        x_lab = rgb2lab(x)
        x_l = x_lab[:,[0,],:,:]
        conv1_2 = self.model1(torch.cat((x_l,torch.zeros_like(x_l),torch.zeros_like(x_l)),dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if(self.classification):
            out_class = self.model_class(conv8_3)
            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())

            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

        return lab2rgb(x_l,out_reg)
# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = ((torch.abs((rgb+.055)/1.055)+eps)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    return out

def xyz2rgb(xyz):

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*((torch.abs(rgb)+eps)**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    return rgb

def xyz2lab(xyz):
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = (torch.abs(xyz_scale)+eps)**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)


    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = ((torch.abs(out)+eps)**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    return out

def rgb2lab(rgb):

    rgb=0.5*(rgb+1.0)
    xyz = rgb2xyz(rgb)
    lab = xyz2lab(xyz)

    l_rs = (lab[:,[0],:,:]-50.)/100.

    ab_rs = lab[:,1:,:,:]/110.
    out = torch.cat((l_rs,ab_rs),dim=1)

    return out

def lab2rgb(x_l, out_reg):

    l = x_l*100. + 50.
    ab = out_reg*110.
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    out = 2*out-1.0

    return out

