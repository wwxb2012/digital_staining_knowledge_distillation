# digital_staining_knowledge_distillation

Here is the repository for our TMI paper: "Digital Staining with Knowledge Distillation: A Unified Framework for Unpaired and Paired-But-Misaligned Data".

# Dataset

DSD_U and DSD_P dataset are placed in ./data/DSD_U and ./data/DSD_P, train/val denotes training set and testing set, A/B/C indicates dark-field unstained images/bright-field H&E stained images/Light enhanced images via histogram matching.

# Python Environment

You need packages in requirements.txt to run our code. 

# Inference

Pretrained models are placed in ./Model/trained .

For inference on DSD_P dataset, you can execute 

`sh DSD_P_inf.sh`

# Training

Pretrained colorization models are places in ./Model/colorization .

For trainging on DSD_P dataset, you can execute 

`sh DSD_P_train.sh`
