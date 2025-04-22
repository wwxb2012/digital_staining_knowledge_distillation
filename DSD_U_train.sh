set -ex
python train.py --config=Yaml/DSD_U.yaml
python test.py --config=Yaml/DSD_U.yaml
