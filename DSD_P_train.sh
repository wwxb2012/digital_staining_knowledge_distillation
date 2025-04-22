set -ex
python train.py --config=Yaml/DSD_P.yaml
python test.py --config=Yaml/DSD_P.yaml
