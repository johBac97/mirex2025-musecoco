import os
import sys
from transformers import MuseCocoModel, MuseCocoConfig

def _main():
    model = MuseCocoModel(MuseCocoConfig())


def _procedures():
    pass

def convert_checkpoint_to_hf():
    '''
    Convert the original fairseq checkpoint to huggingface transformers
    '''
    fairseq_ckpt_fp = '/Users/sonata/Code/StyleTrans/modules/musecoco/checkpoints/large.pt'




if __name__ == '__main__':
    _main()
