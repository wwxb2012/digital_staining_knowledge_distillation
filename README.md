# digital_staining_knowledge_distillation

Here is the repository for our TMI paper: "Digital Staining with Knowledge Distillation: A Unified Framework for Unpaired and Paired-But-Misaligned Data".

# Requirements

Python = 3.9.12
CUDA = 11.6

```Shell
conda create -n digitalstaining python=3.9.21
conda activate digitalstaining
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

The following libraries are also required

```Shell
numpy==1.24.4
opencv-python==4.11.0.86
tqdm==4.67.1
visdom==0.1.8.9
matplotlib==3.9.4
scikit-image==0.24.0
pyyaml==6.0
six==1.17.0
torch-fidelity==0.3.0
pytorch-fid==0.2.1
```

You can install them via
```Shell
pip install -r requirements.txt
```

# Dataset

DSD_U and DSD_P datasets are placed in `./data/DSD_U` and `./data/DSD_P`, train/val denotes the training set and the testing set, A/B/C indicates dark-field unstained images/bright-field H&E stained images/Light enhanced images via histogram matching.


# Inference

Pretrained models are placed in the `./Model/trained` directory.

For inference on DSD_U dataset, you can execute 

`sh DSD_U_inf.sh`

The predicted images are placed in `./output/DSD_U/img_inf` directory.

# Evaluation

For DSD_U, FID, KID and LPIPS can be calculated by 

```Shell
python evaluation/cal_U.py
```

NIQE is calculated by Matlab using [link](https://www.mathworks.com/help/images/ref/niqe.html).

For DSD_P, the matrics can be calculated by:

```Shell
python evaluation/cal_P.py
```

# Training

The pretrained colorization models can be found in the ./Model/colorization directory.

For training on DSD_P dataset, you can execute 

`sh DSD_P_train.sh`

Please adjust the batchSize in the YAML files under ./Yaml according to your device's capabilities.

# Acknowledgements

This project builds upon [Reg-GAN](https://github.com/Kid-Liet/Reg-GAN), and we sincerely thank the original authors for their outstanding work. We are also grateful to the anonymous reviewers for their valuable feedback. Additionally, we acknowledge the contributions of open-source projects such as [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [torch-fidelity](https://github.com/toshas/torch-fidelity), and [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).

