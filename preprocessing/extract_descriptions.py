"""Extract image descriptions from the downloaded files."""
import cPickle as pkl
import glob
import json
import sys
import ipdb
from pathos.multiprocessing import Pool
from unicodedata import normalize
import time
from absl import app
from absl import flags
from tqdm import tqdm
import csv
sys.path.append('../')
from config import TF_MODELS_PATH

sys.path.insert(0, TF_MODELS_PATH + '/research/im2txt/im2txt/data')
from build_mscoco_data import _process_caption

flags.DEFINE_string('data_dir', '../data/GCC/', 'data directory')

FLAGS = flags.FLAGS

def parse_key_words(caption, dic):
  key_words = dic.intersection(caption)
  return key_words

def main(_):
  dict_coco=[]
  for i in open('../data/coco/coco.txt'):
    dict_coco.append(i[:-1])
  dict_coco = set(dict_coco)

  csv.register_dialect('spp',delimiter='\t',quoting=csv.QUOTE_ALL)
  file = csv.reader(open(FLAGS.data_dir+'Train_GCC-training.tsv'),'spp')
  captions = []
  idx=0
  for line in file:
    # start = time.time()
    sen = line[0].decode('ascii','ignore').encode('ascii')
    sen = _process_caption(sen)
    k = parse_key_words(sen, dict_coco)
    if len(k)> 0:
      idx += 1
      print idx
      captions.append(sen)
    # print time.time() - start
  # s = set()
  # files = glob.glob(FLAGS.data_dir + '/*.json')
  # files.sort()
  # for i in tqdm(files):
  #   with open(i, 'r') as g:
  #     data = json.load(g)
  #     for k, v in data.items():
  #       for j in v:
  #         if 'description' in j:
  #           c = normalize('NFKD', j['description']).encode('ascii', 'ignore')
  #           c = c.split('\n')
  #           s.update(c)

  # pool = Pool()
  # captions = pool.map(_process_caption, sens)
  # pool.close()
  # pool.join()
  ipdb.set_trace()
  # There is a sos and eos in each caption, so the actual length is at least 8.
  captions = [i for i in captions if len(i) >= 10]
  print('%s captions parsed' % len(captions))
  with open('../data/GCC/sentences_gcc.pkl', 'w') as f:
    pkl.dump(captions, f)


if __name__ == '__main__':
  app.run(main)
