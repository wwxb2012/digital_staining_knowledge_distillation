import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

concatAC = False
def loadpng(str):
    im = Image.open(str)
    # im = np.array(im)/255.0
    return im
def return_name(str):
    return str.split('/')[-1].split('.')[0]
def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    # print(img.shape,mode)
    if type(img)!=torch.Tensor:
        img = torch.from_numpy(img)
    if mode == 0:
        return img
    elif mode == 1:
        # return img.rot90(0, [1, 2]).flip([1])
        return img.flip([2])
    elif mode == 2:
        return img.flip([1])
    elif mode == 3:
        return img.rot90(2, [1, 2])
    elif mode == 4:
        return img.rot90(1, [1, 2]).flip([1])
    elif mode == 5:
        return img.rot90(0, [1, 2])
    elif mode == 6:
        return img.rot90(1, [1, 2])
    elif mode == 7:
        return img.rot90(2, [1, 2]).flip([1])
class ImageDataset(Dataset):
    def __init__(self, root,count = None,transforms_1=None,transforms_2=None, unaligned=False, size=256, use_edge = False, transforms_edge=None):
        # print('unaligned: ', unaligned)
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)

        self.files_A = sorted(glob.glob("%s/A/*.png" % root))
        self.files_B = sorted(glob.glob("%s/B/*.png" % root))
        if os.path.isdir(os.path.join(root, 'C')):
            self.files_C = sorted(glob.glob("%s/C/*.png" % root))
        
        self.randomfliprotation = True
        self.unaligned = unaligned
        self.size = size
        print('A_size: ', len(self.files_A), 'B_size: ', len(self.files_B))
        
    def __getitem__(self, index):
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        min_length = min(len(self.files_A),len(self.files_B))
        index = index % min_length
        item_A = self.transform1(loadpng(self.files_A[index % len(self.files_A)]))
        if hasattr(self, 'files_C'):
            item_A = np.array(loadpng(self.files_A[index % len(self.files_A)])) #(256,256,3)
            item_C = np.array(loadpng(self.files_C[index % len(self.files_C)]))
            item_A = Image.fromarray(np.uint8(item_A))
            item_C = Image.fromarray(np.uint8(item_C))
            item_A = self.transform1(item_A)
            item_C = self.transform2(item_C)
        if self.unaligned:
            item_B = self.transform2(loadpng(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:  
            item_B = self.transform2(loadpng(self.files_B[index % len(self.files_B)]))
            if item_B.shape!=item_A.shape:
                print(self.files_B[index % len(self.files_B)],self.files_A[index % len(self.files_A)])
                print(item_B.shape,item_A.shape,index)
            assert item_B.shape==item_A.shape
        base_items = {'A': item_A, 'B': item_B}
        if hasattr(self, 'files_C'):
            base_items['C'] = item_C
        if self.randomfliprotation:
            mode = random.randint(0, 7)
            for k in base_items.keys():
                base_items[k]=augment_img(base_items[k], mode=mode)
            if self.unaligned:
                mode = random.randint(0, 7)
                base_items['B'] = augment_img(base_items['B'], mode=mode)
        return base_items
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False, transforms_hdr =None, transforms_val_A =None):
        self.transform = transforms.Compose(transforms_)
        if transforms_val_A==None:
            transforms_val_A = transforms_
        self.transform_A = transforms.Compose(transforms_val_A)
        self.transformhdr = transforms.Compose(transforms_hdr)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*.png" % root))
        self.files_B = sorted(glob.glob("%s/B/*.png" % root))
        if os.path.isdir(os.path.join(root, 'C')):
            self.files_C = sorted(glob.glob("%s/C/*.png" % root))
        print('val_A_size: ', len(self.files_A), 'val_B_size: ', len(self.files_B))

        
    def __getitem__(self, index):
        item_A = self.transform_A(loadpng(self.files_A[index % len(self.files_A)]))
        if self.unaligned:
            item_B = self.transform(loadpng(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(loadpng(self.files_B[index % len(self.files_B)]))
            assert self.files_A[index % len(self.files_A)].split('/')[-1]==self.files_B[index % len(self.files_B)].split('/')[-1]
        base_items = {'A': item_A, 'B': item_B}
        if hasattr(self, 'files_C'):
            item_C = self.transform(loadpng(self.files_C[index % len(self.files_C)]))
            base_items['C'] = item_C
        base_items['name_A']= self.files_A[index % len(self.files_A)]
        return base_items
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))