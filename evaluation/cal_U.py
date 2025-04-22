import os
import glob
import argparse
import subprocess
import numpy as np
import cv2
from natsort import natsorted
if os.path.exists('./tmp'):
    os.system("rm -rf ./tmp")
os.makedirs('./tmp')
if os.path.exists('./tmp2'):
    os.system("rm -rf ./tmp2")
os.makedirs('./tmp2')
parser = argparse.ArgumentParser()
parser.add_argument("--dir_A", type=str, default='./data/DSD_U/train/B')
parser.add_argument("--dir_B", type=str, default='./output/DSD_U/img_inf')
opt = parser.parse_args()

def get_illu_2(rgb):
    pa = (0.114,0.587,0.299) #BGR
    ill = pa[0]*rgb[0]+pa[1]*rgb[1]+pa[2]*rgb[2]
    return ill  
def get_illu(data):
    data1 = np.zeros([data.shape[0],data.shape[1]],dtype = np.float32)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data1[i][j]=get_illu_2(data[i][j])
    return data1
def get_image_illu(data):
    data1 = np.zeros([data.shape[0],data.shape[1],3],dtype = np.float32)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data1[i][j][2]=data1[i][j][1]=data1[i][j][0]=get_illu_2(data[i][j])
    return data1
# opt.dir_A = './data/DSD_U/train/B'

fake_B = True

if not os.path.isdir(opt.dir_B):
    print('dir_B is not a directory')
else:
    img_B_list = glob.glob(os.path.join(opt.dir_B,'*_fake_B.png'))
    if len(img_B_list)==0:
        fake_B = False
        img_B_list = glob.glob(os.path.join(opt.dir_B,'*.png'))
    img_A_list = glob.glob(os.path.join(opt.dir_A,'*.png'))
    if len(img_B_list)==0:
        print('no img in img_B')
    else:
        print('dir_A: ',opt.dir_A)
        print('dir_B: ',opt.dir_B)
        print('number of images in dir_B:  ',len(img_B_list))
        for file in img_B_list:
            os.system("cp "+file+" ./tmp")
            file1=file.split('/')[-1]
            if fake_B:
                os.system("mv "+ "./tmp/"+file1+" ./tmp/"+file1[:-11]+file1[-4:])
        cmd = f'python -W ignore -m pytorch_fid {opt.dir_A} ./tmp --batch-size {min(len(img_B_list),len(img_A_list))}'
        print('------------------ FID Calculation ----------------------')
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        result = str(result)
        pos = result.find('FID:  ')
        result = result[pos+6:]
        result = result.split('n')[0][:-1]
        result0 = float(result)

        print('------------------ KID Calculation ----------------------')
        cmd = f'fidelity --gpu 0 --kid --input1 ./tmp --input2 {opt.dir_A} -b {min(len(img_B_list),len(img_A_list))} --kid-subset-size {min(len(img_B_list),len(img_A_list))}'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        # print(result)
        result = str(result)
        pos = result.find('mean: ')
        result = result[pos+6:]
        result = result.split('k')[0][:-2]
        result1 = float(result)


        print('------------------ LPIPS-c Calculation ----------------------')
        img_B1_list = natsorted(glob.glob(os.path.join('./tmp/','*.png')))
        for file in img_B1_list:

            df_image = cv2.imread(file)
            df_image = get_image_illu(df_image)
            cv2.imwrite(file,df_image,[cv2.IMWRITE_PNG_COMPRESSION,0])
        img_B2_list = natsorted(glob.glob("./data/DSD_U/val/C/*.png"))
        for file in img_B2_list:
            os.system("cp "+file+" ./tmp2")
            file1=file.split('/')[-1]
        
        cmd = f'python -W ignore evaluation/PerceptualSimilarity/lpips_2dirs.py -d0 ./tmp -d1 ./tmp2 -o ./evaluation/PerceptualSimilarity/lpipsoutput.txt --use_gpu'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        result = str(result)
        pos = result.find('ore:  ')
        result = result[pos+6:]
        result = result.split('C')[0][:-4]
        result2 = float(result)
        print(' ')
        print('----------------------------------------------------------')
        print(' ')
        print('FID: ', result0)
        print('KID: ', result1)
        print('LPIPS-c: ', result2)
        os.system("rm -rf ./tmp2")
        os.system("rm -rf ./tmp")