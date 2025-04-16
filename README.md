# digital_staining_knowledge_distillation

Here is the repository for our TMI paper: "Digital Staining with Knowledge Distillation: A Unified Framework for Unpaired and Paired-But-Misaligned Data".

# Dataset

DSD_U and DSD_P datasets are placed in ./data/DSD_U and ./data/DSD_P, train/val denotes the training set and the testing set, A/B/C indicates dark-field unstained images/bright-field H&E stained images/Light enhanced images via histogram matching.

# Python Environment

You need packages in requirements.txt to run our code. 

# Inference

Pretrained models are placed in the ./Model/trained directory.

For inference on DSD_P dataset, you can execute 

`sh DSD_P_inf.sh`

# Training

The pretrained colorization models can be found in the ./Model/colorization directory.

For training on DSD_P dataset, you can execute 

`sh DSD_P_train.sh`

Please adjust the batchSize in the YAML files under ./Yaml according to your device's capabilities.
