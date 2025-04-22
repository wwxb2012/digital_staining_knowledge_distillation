import os, time, random, cv2
import numpy as np
from glob import glob
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from natsort import natsorted
import subprocess

def load_images(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32")
    return img

def calculate_psnr_np(img1, img2):
    SE_map = (1.*img1-img2)**2
    cur_MSE = np.mean(SE_map)
    return 20*np.log10(255./np.sqrt(cur_MSE))


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return new_ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(new_ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return new_ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def new_ssim(img1, img2):
    C1 = (0.01 * 255.0)**2
    C2 = (0.03 * 255.0)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
output_image = []
gt_image = []
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir_A", type=str, default="./data/DSD_P/val/B")
parser.add_argument("--dir_B", type=str, default="./output/DSD_P/img_inf")
opt = parser.parse_args()
qwq = opt.dir_B
if not os.path.isdir(opt.dir_B):
    print('dir_B is not a directory')
# 
fake_B = True

gt_path = opt.dir_A
print('gt_path: ', gt_path)
print('path_to_judge: ', qwq)
output_image_names = glob(qwq+'/*_fake_B.png')
if len(output_image_names)==0:
    fake_B = False
    output_image_names = glob(qwq+'/*.png')
gt_image_names = glob(gt_path+'/*.png')
output_image_names=natsorted(output_image_names)
gt_image_names=natsorted(gt_image_names)
assert len(output_image_names)==len(gt_image_names)
for idx in range(len(output_image_names)):
    output_image.append(load_images(output_image_names[idx]))
    gt_image.append(load_images(gt_image_names[idx]))
psnr = []
ssim = []
for idx in range(len(output_image_names)):
    psnr1 = calculate_psnr_np(output_image[idx],gt_image[idx])
    ssim2 = calculate_ssim(output_image[idx],gt_image[idx])
    psnr.append(psnr1)
    ssim.append(ssim2)
    # print(output_image_names[idx].split('/')[-1],psnr1,ssim1)
print('PSNR:  ',np.mean(np.array(psnr)))
print('SSIM:  ',np.mean(np.array(ssim)))
if os.path.exists('./tmp'):
    os.system("rm -rf ./tmp")
os.makedirs('./tmp')
for file in output_image_names:
    os.system("cp "+file+" ./tmp")
    file1=file.split('/')[-1]
    if fake_B:
        os.system("mv "+ "./tmp/"+file1+" ./tmp/"+file1[:-11]+file1[-4:])
cmd = f'python -W ignore evaluation/PerceptualSimilarity/lpips_2dirs.py -d0 ./tmp -d1 {gt_path} -o ./evaluation/PerceptualSimilarity/lpipsoutput.txt --use_gpu'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
result = str(result)
pos = result.find('ore:  ')
result = result[pos+6:]
result = result.split('C')[0][:-4]
result = float(result)
print('LPIPS: ', result)
os.system("rm -rf ./tmp")
