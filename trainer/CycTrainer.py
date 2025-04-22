#!/usr/bin/python3
import itertools
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import Logger,ReplayBuffer
from .datasets import ImageDataset,ValDataset
from Model.CycleGan import *
from .utils import ToTensor,smooothing_loss
from .reg import Reg
from torchvision.transforms import RandomAffine
from .transformer import Transformer_2D
import numpy as np
import cv2
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore')
from tqdm import tqdm
import time




def get_lr(current_epoch, total_epoch, setup_lr):
    if current_epoch<=int(0.667*total_epoch):
        return setup_lr
    else:
        ratio = -3*(current_epoch / total_epoch)+3
        return ratio*setup_lr

    
def create_dir(name): 
    try:
        os.mkdir(name)
    except:
        pass

def adjust_keys(nett):
    return OrderedDict((k[7:], v) for k, v in nett.items())


class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_lr = config['lr']
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc'],ndf=64).cuda()
        self.netD_B = Discriminator(config['output_nc'],ndf=64).cuda()

        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['output_nc'],config['output_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            ndf = 64
            if 'use_iedf' in self.config and self.config['use_iedf']:
                self.netG_B2A = Generator(config['output_nc'], min(config['output_nc'],config['input_nc']),ndf=ndf).cuda()
            else:
                self.netG_B2A = Generator(config['output_nc'], config['input_nc'],ndf=ndf).cuda()
            if 'half_cycle' in config and config['half_cycle']:
                self.netD_A = Discriminator(min(config['input_nc'],config['output_nc']),ndf=ndf).cuda()
            else:
                self.netD_A = Discriminator(config['input_nc'],ndf=ndf).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if 'distillation' in self.config:
            if config['distillation']:
                self.netT = SIGGRAPHGenerator(config['output_nc'],config['output_nc']).cuda()
                assert 'teacher_model_location' in self.config
                self.netT.load_state_dict(adjust_keys(torch.load(self.config['teacher_model_location'])))
                self.netT.eval()
                for param in self.netT.parameters():
                    param.requires_grad = False

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.smoothL1 = torch.nn.SmoothL1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_C = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(config['batchSize'],1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(config['batchSize'],1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        level = config['noise_level']  # set noise level
        transforms_1 = [#ToPILImage(),
                        RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fill=0),
                        ToTensor(),
                        ]
    
        transforms_2 = [#ToPILImage(),
                        RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fill=0),
                        ToTensor()
                        ]

        unaligned_train = False
        if 'unaligned' in config:
            unaligned_train = config['unaligned']
        self.dataloader = DataLoader(ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2, unaligned=unaligned_train, size=config['size']),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [
                          ToTensor()
                         ]
        
        # if 'use_zscore' in self.config and self.config['use_zscore']:
        #     transforms_val_A = val_transforms.copy()
        #     if 'DSD_U' in self.config['dataroot']:
        #         transforms_1.append( Normalize((-0.9364,-0.9261,-0.8997),(0.05559,0.05285,0.06604)))
        #         transforms_val_A.append(Normalize((-0.9364,-0.9261,-0.8997),(0.05559,0.05285,0.06604)))
        #     else:
        #         transforms_1.append(Normalize((-0.5330, -0.4736,-0.2095),(0.16726,0.17496,0.22668)))
        #         transforms_val_A.append(Normalize((-0.5330, -0.4736,-0.2095),(0.16726,0.17496,0.22668)))
        #     self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False,transforms_val_A = transforms_val_A),
        #                         batch_size=1, shuffle=False, num_workers=config['n_cpu'])
            
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False),
                                batch_size=1, shuffle=False, num_workers=config['n_cpu'])

 
        # Loss plot -- logger via visdom
        # self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))   

        self.val_dataset = ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False)
        self.criterionL1 = torch.nn.L1Loss()



        
    def train(self):

        if self.config['epoch']>0:
            epoch = self.config['epoch']
            if os.path.getsize(self.config['save_root'] +str(epoch) +'_netG_A2B.pth')==0:
                print(epoch+'\' s model doesn\'t exists')
                self.config['epoch'] = 0
            else:
                self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] +str(epoch) +'_netG_A2B.pth'))
                if self.config['regist'] and os.path.exists(self.config['save_root']+str(epoch)+'_Regist.pth'):
                    self.R_A.load_state_dict(torch.load(self.config['save_root']+str(epoch)+'_Regist.pth'))
                self.netD_B.load_state_dict(torch.load(self.config['save_root'] +str(epoch) +'netD_B_3D.pth'))
                if self.config['bidirect'] and os.path.exists(self.config['save_root']+str(epoch)+'_netG_B2A.pth'):
                    self.netG_B2A.load_state_dict(torch.load(self.config['save_root'] +str(epoch) +'_netG_B2A.pth'))
                    self.netD_A.load_state_dict(torch.load(self.config['save_root'] +str(epoch) +'_netD_A.pth'))


        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in tqdm(enumerate(self.dataloader)):
                if 'use_iedf' in self.config  and self.config['use_iedf']:
                    real_A_0 = Variable(self.input_A.resize_(batch['A'].size()).copy_(batch['A']))
                    real_A_1 = Variable(self.input_C.resize_(batch['C'].size()).copy_(batch['C']))
                    if real_A_0.shape[1]==1:
                        real_A_0 = torch.cat((real_A_0,real_A_0,real_A_0),dim=1)
                    if real_A_1.shape[1]==3:
                        real_A_1 = (real_A_1[:,0]+real_A_1[:,1]+real_A_1[:,2])/3.0
                        real_A_1 = real_A_1.unsqueeze(1)
                    real_A = torch.cat((real_A_0,real_A_1),dim=1)
                else:
                    real_A = Variable(self.input_A.resize_(batch['A'].size()).copy_(batch['A']))
                real_B = Variable(self.input_B.resize_(batch['B'].size()).copy_(batch['B']))
                if 'C' in batch:
                    real_C = Variable(self.input_C.resize_(batch['C'].size()).copy_(batch['C']))
                    if real_C.shape[1]==1:
                        real_C = torch.cat((real_C,real_C,real_C),dim=1)
                if 'distillation' in self.config:
                    if self.config['distillation']:
                        pretrained_fake_B = self.netT(real_C)
                        # if 'remove_ie' in self.config and self.config['remove_ie']:
                        #     pretrained_fake_B = self.netT(real_A)
                if self.config['bidirect']:   # C dir
                    if self.config['regist']:    #C + R
                        if 'half_cycle' in self.config and self.config['half_cycle']:
                            self.optimizer_R_A.zero_grad()
                            self.optimizer_G.zero_grad()
                            # GAN loss
                            if self.target_real.shape[0]!=real_A.shape[0]:
                                Tensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
                                self.target_real = Variable(Tensor(real_A.shape[0],1).fill_(1.0), requires_grad=False)
                                self.target_fake = Variable(Tensor(real_A.shape[0],1).fill_(0.0), requires_grad=False)
                            fake_B = self.netG_A2B(real_A)
                            pred_fake = self.netD_B(fake_B)
                            loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                            real_A = real_A[:,:3]
                            Trans = self.R_A(fake_B,real_B) 
                            SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                            SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                            SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                            
                            # Cycle loss
                            recovered_A = self.netG_B2A(fake_B)
                            loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A[:,:3])

                            pred_fake = self.netD_A(recovered_A)
                            loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                            loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA  + SR_loss +SM_loss
                            if 'distillation' in self.config:
                                if self.config['distillation']:
                                    distillation_loss = self.config['distillation']*self.criterionL1(pretrained_fake_B, fake_B)
                                    loss_Total+=distillation_loss

                            loss_Total.backward()
                            self.optimizer_G.step()
                            self.optimizer_R_A.step()
                            # torch.cuda.empty_cache()
                            
                            ###### Discriminator A ######
                            self.optimizer_D_A.zero_grad()
                            # Real loss
                            pred_real = self.netD_A(real_A[:,:3])
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                            # Fake loss
                            fake_A = self.fake_A_buffer.push_and_pop(recovered_A)
                            pred_fake = self.netD_A(fake_A.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_A = (loss_D_real + loss_D_fake)

                            loss_D_A.backward()
                            self.optimizer_D_A.step()

                            ###################################
                            # torch.cuda.empty_cache()

                            ###### Discriminator B ######
                            self.optimizer_D_B.zero_grad()

                            # Real loss
                            pred_real = self.netD_B(real_B)
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                            # Fake loss
                            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                            pred_fake = self.netD_B(fake_B.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_B = (loss_D_real + loss_D_fake)
                            loss_D_B.backward()
                            self.optimizer_D_B.step()
                        else:
                            self.optimizer_R_A.zero_grad()
                            self.optimizer_G.zero_grad()
                            # GAN loss
                            if self.target_real.shape[0]!=real_A.shape[0]:
                                Tensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
                                self.target_real = Variable(Tensor(real_A.shape[0],1).fill_(1.0), requires_grad=False)
                                self.target_fake = Variable(Tensor(real_A.shape[0],1).fill_(0.0), requires_grad=False)
                            fake_B = self.netG_A2B(real_A)
                            pred_fake = self.netD_B(fake_B)
                            loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                            
                            fake_A = self.netG_B2A(real_B)
                            pred_fake = self.netD_A(fake_A)
                            loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                            Trans = self.R_A(fake_B,real_B) 
                            # print(Trans.shape)
                            SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                            SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                            SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                            
                            # Cycle loss
                            recovered_A = self.netG_B2A(fake_B)
                            loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                            recovered_B = self.netG_A2B(fake_A)
                            loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                            # Total loss
                            loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss +SM_loss
                            if 'distillation' in self.config:
                                if self.config['distillation']:
                                    distillation_loss = self.config['distillation']*self.criterionL1(pretrained_fake_B, fake_B)
                                    loss_Total+=distillation_loss
                            loss_Total.backward()
                            self.optimizer_G.step()
                            self.optimizer_R_A.step()
                            # torch.cuda.empty_cache()
                            
                            ###### Discriminator A ######
                            self.optimizer_D_A.zero_grad()
                            # Real loss
                            pred_real = self.netD_A(real_A)
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                            # Fake loss
                            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                            pred_fake = self.netD_A(fake_A.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_A = (loss_D_real + loss_D_fake)
                            loss_D_A.backward()
                            self.optimizer_D_A.step()
                            ###################################
                            # torch.cuda.empty_cache()

                            ###### Discriminator B ######
                            self.optimizer_D_B.zero_grad()

                            # Real loss
                            pred_real = self.netD_B(real_B)
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                            # Fake loss
                            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                            pred_fake = self.netD_B(fake_B.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_B = (loss_D_real + loss_D_fake)
                            loss_D_B.backward()
                            self.optimizer_D_B.step()
                            # torch.cuda.empty_cache()
                    
                    else: #only  dir:  C
                        if 'half_cycle' in self.config and self.config['half_cycle']:
                            self.optimizer_G.zero_grad()
                            # GAN loss
                            if self.target_real.shape[0]!=real_A.shape[0]:
                                Tensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
                                self.target_real = Variable(Tensor(real_A.shape[0],1).fill_(1.0), requires_grad=False)
                                self.target_fake = Variable(Tensor(real_A.shape[0],1).fill_(0.0), requires_grad=False)
                            
                            fake_B = self.netG_A2B(real_A)
                            real_A = real_A[:,:3]
                            pred_fake = self.netD_B(fake_B)
                            loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                            # Cycle loss
                            recovered_A = self.netG_B2A(fake_B)
                            loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A[:,:3])

                            pred_fake = self.netD_A(recovered_A)
                            loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)


                            # Total loss
                            loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA
                            if 'distillation' in self.config:
                                if self.config['distillation']:
                                    distillation_loss = self.config['distillation']*self.criterionL1(pretrained_fake_B, fake_B)
                                    loss_Total+=distillation_loss
                            loss_Total.backward()
                            self.optimizer_G.step()

                            ###### Discriminator A ######
                            self.optimizer_D_A.zero_grad()
                            # Real loss
                            pred_real = self.netD_A(real_A)
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                            # Fake loss
                            fake_A = self.fake_A_buffer.push_and_pop(recovered_A)
                            pred_fake = self.netD_A(fake_A.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_A = (loss_D_real + loss_D_fake)
                            loss_D_A.backward()

                            self.optimizer_D_A.step()
                            ###################################

                            ###### Discriminator B ######
                            self.optimizer_D_B.zero_grad()

                            # Real loss
                            pred_real = self.netD_B(real_B)
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                            # Fake loss
                            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                            pred_fake = self.netD_B(fake_B.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_B = (loss_D_real + loss_D_fake)
                            loss_D_B.backward()

                            self.optimizer_D_B.step()
                            ###################################
                        else:
                            self.optimizer_G.zero_grad()
                            # GAN loss
                            if self.target_real.shape[0]!=real_A.shape[0]:
                                Tensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
                                self.target_real = Variable(Tensor(real_A.shape[0],1).fill_(1.0), requires_grad=False)
                                self.target_fake = Variable(Tensor(real_A.shape[0],1).fill_(0.0), requires_grad=False)
                            
                            fake_B = self.netG_A2B(real_A)
                            pred_fake = self.netD_B(fake_B)
                            loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                            fake_A = self.netG_B2A(real_B)
                            pred_fake = self.netD_A(fake_A)
                            loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                            # Cycle loss
                            recovered_A = self.netG_B2A(fake_B)
                            loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                            recovered_B = self.netG_A2B(fake_A)
                            loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                            # Total loss
                            loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                            if 'distillation' in self.config:
                                if self.config['distillation']:
                                    distillation_loss = self.config['distillation']*self.criterionL1(pretrained_fake_B, fake_B)
                                    loss_Total+=distillation_loss
                            loss_Total.backward()
                            self.optimizer_G.step()

                            ###### Discriminator A ######
                            self.optimizer_D_A.zero_grad()
                            # Real loss
                            pred_real = self.netD_A(real_A)
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                            # Fake loss
                            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                            pred_fake = self.netD_A(fake_A.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_A = (loss_D_real + loss_D_fake)
                            loss_D_A.backward()

                            self.optimizer_D_A.step()
                            ###################################

                            ###### Discriminator B ######
                            self.optimizer_D_B.zero_grad()

                            # Real loss
                            pred_real = self.netD_B(real_B)
                            loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                            # Fake loss
                            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                            pred_fake = self.netD_B(fake_B.detach())
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                            # Total loss
                            loss_D_B = (loss_D_real + loss_D_fake)
                            loss_D_B.backward()

                            self.optimizer_D_B.step()
                            ###################################
                        
                        
                        
                else:                  # s dir :NC
                    if self.config['regist']:    # NC+R
                        if self.target_real.shape[0]!=real_A.shape[0]:
                            Tensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
                            self.target_real = Variable(Tensor(real_A.shape[0],1).fill_(1.0), requires_grad=False)
                            self.target_fake = Variable(Tensor(real_A.shape[0],1).fill_(0.0), requires_grad=False)
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss

                        fake_B = self.netG_A2B(real_A)
                        if 'half_regist' in self.config and self.config['half_regist'] and epoch>self.config['n_epochs']:
                            Trans = self.R_A(fake_B, torch.zeros_like(real_B).to('cuda'))
                        else:
                            Trans = self.R_A(fake_B,real_B) 
                        
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        loss_G_A = 0
                        adv_loss += self.config['Adv_lamda'] *loss_G_A
                        toal_loss = SM_loss+adv_loss+SR_loss
                        if 'distillation' in self.config:
                            if self.config['distillation']:
                                distillation_loss = self.config['distillation']*self.criterionL1(pretrained_fake_B, fake_B)
                                toal_loss+=distillation_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        loss_D_B.backward()
                        self.optimizer_D_B.step()
    
                        
                        
                        
                    else:        # only NC
                        if self.target_real.shape[0]!=real_A.shape[0]:
                            Tensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
                            self.target_real = Variable(Tensor(real_A.shape[0],1).fill_(1.0), requires_grad=False)
                            self.target_fake = Variable(Tensor(real_A.shape[0],1).fill_(0.0), requires_grad=False)
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        fake_B_original = fake_B.clone().detach()
                        #### GAN aligin loss
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(fake_B,real_B)###SR
                        toal_loss = adv_loss+SR_loss
                        if 'distillation' in self.config:
                            if self.config['distillation']:
                                distillation_loss = self.config['distillation']*self.criterionL1(pretrained_fake_B, fake_B)
                                toal_loss+=distillation_loss
                        toal_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################


                # self.logger.log({'loss_D_B': loss_D_B,},
                #        images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})#,'SR':SysRegist_A2B

    #         # Save models checkpoints
            create_dir(self.config["save_root"])
            if (epoch+1) % 10 ==0: #You can replace with 20 or 50 to save storage
                torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + str(epoch+1)+'_netG_A2B.pth')
                if self.config['regist']:
                    torch.save(self.R_A.state_dict(), self.config['save_root']  + str(epoch+1)+'_Regist.pth')
                if self.config['bidirect']:
                    torch.save(self.netG_B2A.state_dict(), self.config['save_root']  + str(epoch+1)+'_netG_B2A.pth')
                    torch.save(self.netD_A.state_dict(), self.config['save_root']  + str(epoch+1)+'_netD_A.pth')

                torch.save(self.netD_B.state_dict(), self.config['save_root']  + str(epoch+1)+'netD_B_3D.pth')


            if epoch> 200:
                self.current_lr = get_lr(epoch, self.config['n_epochs'], self.config['lr'])
                for param_group in self.optimizer_G.param_groups:
                    param_group['lr']=self.current_lr
                for param_group in self.optimizer_D_B.param_groups:
                    param_group['lr']=self.current_lr
                if self.config['regist']:
                    for param_group in self.optimizer_R_A.param_groups:
                        param_group['lr']=self.current_lr
                if self.config['bidirect']:
                    for param_group in self.optimizer_D_A.param_groups:
                        param_group['lr']=self.current_lr

 
                


    def save_img(self,config, fake_B,name_A,epoch):
        im = fake_B
        im = im*127.5+127.5
        if len(im.shape)==2:
            im = np.stack([im,im,im],axis=0)
        im = np.transpose(im[::-1],(1,2,0))
        cv2.imwrite(config['save_root']+'img/'+str(epoch)+'/'+name_A.split('/')[-1].split('.')[0]+'_fake_B.png',im,[cv2.IMWRITE_PNG_COMPRESSION,0])
        return
    def save_img2(self,config, fake_B,name_A,epoch):
        im = fake_B
        im = im*127.5+127.5
        if len(im.shape)==2:
            im = np.stack([im,im,im],axis=0)
        im = np.transpose(im[::-1],(1,2,0))
        cv2.imwrite(config['image_save']+'/'+name_A.split('/')[-1].split('.')[0]+'_fake_B.png',im,[cv2.IMWRITE_PNG_COMPRESSION,0])
        return
        
    def test(self):
        epoch = self.config['epoch_to_test']
        create_dir(self.config['save_root']+'img/'+str(epoch))
        if 'inf_model_path' in self.config and os.path.exists(self.config['inf_model_path']):
            self.netG_A2B.load_state_dict(torch.load(self.config['inf_model_path']))
        else:
            self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] +str(epoch) +'_netG_A2B.pth'))

        print('model route:  ',self.config['save_root'])
        print('save img root: ', self.config['image_save'])
        print('Test epoch number:  ', epoch)
        create_dir(self.config['image_save'])
        start_time = time.time()
        total_inf_time = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_data)):
                if 'use_iedf' in self.config  and self.config['use_iedf']:
                    real_A_0 = Variable(self.input_A.resize_(batch['A'].size()).copy_(batch['A']))
                    if real_A_0.shape[1]==1:
                        real_A_0 = torch.cat((real_A_0,real_A_0,real_A_0),dim=1)
                    real_A_1 = Variable(self.input_C.resize_(batch['C'].size()).copy_(batch['C']))
                    if real_A_1.shape[1]==3:
                        real_A_1 = (real_A_1[:,0]+real_A_1[:,1]+real_A_1[:,2])/3.0
                        real_A_1 = real_A_1.unsqueeze(1)
                    real_A = torch.cat((real_A_0,real_A_1),dim=1)
                else:
                    real_A = Variable(batch['A'].clone().to(torch.device('cuda')))
                if self.config['input_nc']==1 and real_A.shape[1]>1:
                    real_A = real_A[:,0]
                    real_A = real_A.unsqueeze(1)
                real_B = Variable(batch['B'].clone().to(torch.device('cuda')))
                if self.config['output_nc']==1 and real_B.shape[1]>1:
                    real_B = real_B[:,0]
                    real_B = real_B.unsqueeze(1)
                name_A = batch['name_A'][0]
                fake_B = self.netG_A2B(real_A)
                # total_inf_time += time.time() - start_time_inf
                self.save_img2(self.config,fake_B.detach().cpu().numpy().squeeze(),name_A,epoch)   
            # print("--- Total time: %s seconds ---" % (time.time()-start_time))
            # print("--- Total inf time: %s seconds ---" % (total_inf_time))
            