#!/usr/bin/python3

import argparse
import os
from trainer import Cyc_Trainer,Nice_Trainer,P2p_Trainer,Munit_Trainer,Unit_Trainer
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    elif config['name'] == 'Munit':
        trainer = Munit_Trainer(config)
    elif config['name'] == 'Unit':
        trainer = Unit_Trainer(config)
    elif config['name'] == 'NiceGAN':
        trainer = Nice_Trainer(config)
    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)

    trainer.train()
    
    



###################################
if __name__ == '__main__':
    main()