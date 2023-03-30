import os

from absl import flags

# coco caption + oidv4
flags.DEFINE_integer('vocab_size', 18680, 'vocab size')
flags.DEFINE_integer('pad_id', 0, 'PAD')
flags.DEFINE_integer('start_id', 1, 'SOS')
flags.DEFINE_integer('end_id', 2, 'EOS')
NUM_DESCRIPTIONS = 2322635

TF_MODELS_PATH = '/data/songpeipei/conda2/envs/tf12py2/lib/python2.7/site-packages/'
COCO_PATH = '/data/songpeipei/OpenData/coco-caption-master'

