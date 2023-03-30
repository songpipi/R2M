"""Train object-to-sentence model.

python initialization/obj2sen.py --batch_size 512 --save_checkpoint_steps 5000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import ipdb
from config import NUM_DESCRIPTIONS,TF_MODELS_PATH
AUTOTUNE = tf.contrib.data.AUTOTUNE

sys.path.append('../')
from misc_fn import crop_sentence
from misc_fn import transform_grads_fn
from misc_fn import validate_batch_size_for_multi_gpu
from misc_fn import get_len

sys.path.append(TF_MODELS_PATH + 'tensorflow/models-master/research/slim')
from nets import inception_v4
sys.path.append(TF_MODELS_PATH + 'sonnet')
from sonnet.python.modules import relational_memory_wei as relational_memory

tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer('intra_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_integer('inter_op_parallelism_threads', 0, 'Number of threads')

tf.flags.DEFINE_bool('multi_gpu', False, 'use multi gpus')

tf.flags.DEFINE_integer('emb_dim', 512, 'emb dim')

tf.flags.DEFINE_integer('mem_dim', 512, 'mem dim')

tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')

tf.flags.DEFINE_string('job_dir', 'rebuttal/end2end', 'job dir')

tf.flags.DEFINE_integer('head', 2, 'head num')

tf.flags.DEFINE_integer('size', 256, 'head size')

tf.flags.DEFINE_string('model', 'no', 'job dir')

tf.flags.DEFINE_integer('batch_size', 256, 'batch size')

tf.flags.DEFINE_integer('pad_length', 20, 'pad len')

tf.flags.DEFINE_integer('max_steps', 10000, 'training steps')

tf.flags.DEFINE_float('weight_decay', 0, 'weight decay')

tf.flags.DEFINE_float('lr', 0.0001, 'learning rate')

tf.flags.DEFINE_float('w_im', 1, 'loss weight')

tf.flags.DEFINE_string('inc_ckpt', '../data/inception_v4.ckpt', 'InceptionV4 checkpoint')

tf.flags.DEFINE_integer('save_summary_steps', 100, 'save summary steps')

tf.flags.DEFINE_integer('save_checkpoint_steps', 1000, 'save ckpt')

FLAGS = tf.flags.FLAGS

def crit(im,s):
  # compute image-sentence score matrix
  margin = 0.2
  # Cosine similarity between all the image and sentence pairs
  scores = tf.matmul(im,s,transpose_b = True)
  diagonal = tf.reshape(tf.diag_part(scores),[tf.shape(im)[0],1])
  d1 = tf.broadcast_to(diagonal,tf.shape(scores))
  d2 = tf.broadcast_to(tf.transpose(diagonal,[1,0]),tf.shape(scores))

  # compare every diagonal score to scores in its column
  # caption retrieval
  cost_s = tf.clip_by_value((margin + scores - d1),clip_value_min=0,clip_value_max=10000) #(b,b)
  # compare every diagonal score to scores in its row
  # image retrieval
  cost_im = tf.clip_by_value((margin + scores - d2),clip_value_min=0,clip_value_max=10000)
  # keep the maximum violating negative for each query
  cost_s = tf.reduce_max(cost_s, axis = 0)
  cost_im = tf.reduce_max(cost_im, axis = 1)
  loss_m = cost_s + cost_im

  return tf.reduce_mean(loss_m)

def inner_rec(features,params,is_training):
  im = features['im']
  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    net, _ = inception_v4.inception_v4(im, None, is_training=False)
  net = tf.squeeze(net, [1, 2])
  feat = slim.fully_connected(net, FLAGS.mem_dim, activation_fn=None) 
  feat = tf.nn.l2_normalize(feat, axis=1)
  inc_saver = tf.train.Saver(tf.global_variables('InceptionV4'))

  with tf.variable_scope('Discriminator', reuse=True):
    embedding = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))

    key, lk = features['classes'], features['num']
    key = tf.nn.embedding_lookup(embedding, key)

    cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim)
    if is_training:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, params.keep_prob,
                                           params.keep_prob)
    out, initial_state = tf.nn.dynamic_rnn(cell, key, lk, dtype=tf.float32)

  feat_obj = tf.nn.l2_normalize(initial_state[1], axis=1)
  batch_size = tf.shape(feat_obj)[0]

  with tf.variable_scope('Generator', reuse=True):
    w = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))
    softmax_w = tf.matrix_transpose(w)
    softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])
    
    if FLAGS.model == 'abla_lstm_dec' :
      cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim,reuse = tf.AUTO_REUSE)
      if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, params.keep_prob, params.keep_prob)
    else:
      cell = relational_memory.RelationalMemory(mem_slots=1, head_size=FLAGS.size, num_heads=FLAGS.head)
    attention = relational_memory.MultiHeadAttention_dec(head_size=FLAGS.size, num_heads=FLAGS.head)

    state = cell.zero_state(batch_size, tf.float32)
 
    rnn_outs, sequence = [],[]
    tf.get_variable_scope().reuse_variables()

    for t in range(FLAGS.pad_length):
      if t == 0: 
        rnn_inp = tf.zeros([batch_size], tf.int32) + FLAGS.start_id
      rnn_inp = tf.nn.embedding_lookup(embedding, rnn_inp) 
      if FLAGS.model == 'abla_wo_fmc':
        inputs = tf.concat([rnn_inp,feat_obj],axis=1)
      else:
        inputs,_ = attention.multihead_attention(tf.stack([rnn_inp,feat_obj],axis=1))
      if FLAGS.model == 'abla_lstm_dec' :
        inputs = tf.contrib.layers.fully_connected(tf.reshape(inputs,[-1,2*FLAGS.mem_dim]), FLAGS.mem_dim, \
        activation_fn=None, reuse=tf.AUTO_REUSE,scope ='fusion')
        out, state = cell(inputs,state) 
      else:     
        out, state,_ = cell(inputs, state)
      rnn_outs.append(out)
      logit = tf.nn.bias_add(tf.matmul(out, softmax_w), softmax_b)
      fake = tf.argmax(logit, axis=1, output_type=tf.int32)
      sequence.append(fake)
      rnn_inp = fake
    rnn_out = tf.stack(rnn_outs,axis=1)
    sequence = tf.stack(sequence, axis=1)

  with tf.variable_scope('Reconstructor', reuse=True):
    if FLAGS.model == 'abla_lstm_rec' :
      cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim,reuse = tf.AUTO_REUSE)
      if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, params.keep_prob, params.keep_prob)    
    else:
      cell = relational_memory.RelationalMemory(mem_slots=1, head_size=FLAGS.size, num_heads=FLAGS.head)    
    state = cell.zero_state(batch_size, tf.float32)
    rnn_outs = []
    tf.get_variable_scope().reuse_variables()
    for t in range(FLAGS.pad_length):
      if FLAGS.model == 'abla_lstm_rec' :
        out, state = cell(rnn_out[:,t,:], state)
      else:
        out, state,_ = cell(rnn_out[:,t,:], state)
      rnn_outs.append(out)
  rnn_out = tf.stack(rnn_outs,axis=1)
  
  length = get_len(sequence, FLAGS.end_id)
  idx = tf.transpose(tf.stack([tf.range(tf.shape(length)[0]), length - 1]))
  rec_feat = tf.gather_nd(rnn_out, idx)

  loss_rec = tf.losses.mean_squared_error(feat_obj,rec_feat)
  loss_inner = crit(feat,rec_feat)
  loss = FLAGS.w_im*loss_rec + loss_inner
  return loss,inc_saver

def model_fn(features, labels, mode, params):
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  with tf.variable_scope('Discriminator'):
    embedding = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))

    key, lk = features['key'], features['len']
    key = tf.nn.embedding_lookup(embedding, key)
    sentence, ls = labels['sentence'], labels['len']
    targets = sentence[:, 1:FLAGS.pad_length+1]
    sentence = sentence[:, :FLAGS.pad_length]
    ls -= 1
    sentence = tf.nn.embedding_lookup(embedding, sentence)

    cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim)
    if is_training:
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, params.keep_prob,
                                           params.keep_prob)
    out, initial_state = tf.nn.dynamic_rnn(cell, key, lk, dtype=tf.float32)

  feat = tf.nn.l2_normalize(initial_state[1], axis=1)
  batch_size = tf.shape(feat)[0]

  with tf.variable_scope('Generator'):
    embedding = tf.get_variable(
      name='embedding',
      shape=[FLAGS.vocab_size, FLAGS.emb_dim],
      initializer=tf.random_uniform_initializer(-0.08, 0.08))
    softmax_w = tf.matrix_transpose(embedding)
    softmax_b = tf.get_variable('softmax_b', [FLAGS.vocab_size])

    # cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim,reuse = tf.AUTO_REUSE)
    # if is_training:
    #   cell = tf.nn.rnn_cell.DropoutWrapper(cell, params.keep_prob, params.keep_prob)

    cell = relational_memory.RelationalMemory(mem_slots=1, head_size=FLAGS.size, num_heads=FLAGS.head)
    attention = relational_memory.MultiHeadAttention_dec(head_size=FLAGS.size, num_heads=FLAGS.head)

    state = cell.zero_state(batch_size, tf.float32)
 
    rnn_outs = []
    tf.get_variable_scope().reuse_variables()
    for t in range(FLAGS.pad_length):
      inputs,_ = attention.multihead_attention(tf.stack([sentence[:,t,:],feat],axis=1))
      # inputs = tf.concat([sentence[:,t,:],feat],axis=1)
      # inputs = tf.contrib.layers.fully_connected(tf.reshape(inputs,[-1,2*FLAGS.mem_dim]), FLAGS.mem_dim, activation_fn=None, reuse=tf.AUTO_REUSE,scope ='fusion')
      # out, state = cell(inputs,state)
      out, state,_ = cell(inputs, state)
      rnn_outs.append(out)
    rnn_out = tf.stack(rnn_outs,axis=1)
    out = tf.reshape(rnn_out, [-1, FLAGS.mem_dim])
    logits = tf.nn.bias_add(tf.matmul(out, softmax_w), softmax_b)
    logits = tf.reshape(logits, [batch_size, -1, FLAGS.vocab_size])
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

  mask = tf.sequence_mask(ls, tf.shape(sentence)[1])
  targets = tf.boolean_mask(targets, mask)
  logits = tf.boolean_mask(logits, mask)
  loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                        logits=logits)
  loss1 = tf.reduce_mean(loss1)

  with tf.variable_scope('Reconstructor'):
    # cell = tf.nn.rnn_cell.BasicLSTMCell(params.mem_dim,reuse = tf.AUTO_REUSE)
    # if is_training:
    #   cell = tf.nn.rnn_cell.DropoutWrapper(cell, params.keep_prob, params.keep_prob)

    cell = relational_memory.RelationalMemory(mem_slots=1, head_size=FLAGS.size, num_heads=FLAGS.head)
    state = cell.zero_state(batch_size, tf.float32)
    rnn_outs = []
    tf.get_variable_scope().reuse_variables()
    for t in range(FLAGS.pad_length):
      out, state,_ = cell(rnn_out[:,t,:], state)
      # out, state = cell(rnn_out[:,t,:], state)
      rnn_outs.append(out)
  rnn_out = tf.stack(rnn_outs,axis=1)


  idx = tf.transpose(tf.stack([tf.range(tf.shape(ls)[0]), ls - 1]))
  rec_feat = tf.gather_nd(rnn_out, idx)  
  loss2 = tf.losses.mean_squared_error(feat,rec_feat)

  loss = loss1 + FLAGS.w_im*loss2

  loss_im,inc_saver = inner_rec(features,params,is_training)

  loss = loss+loss_im

  train_var = []
  train_var.extend(tf.trainable_variables('Discriminator'))
  train_var.extend(tf.trainable_variables('Generator'))
  train_var.extend(tf.trainable_variables('Reconstructor'))

  opt = tf.train.AdamOptimizer(params.lr)
  if params.multi_gpu:
    opt = tf.contrib.estimator.TowerOptimizer(opt)
  grads = opt.compute_gradients(loss,train_var)
  grads = transform_grads_fn(grads)
  train_op = opt.apply_gradients(grads, global_step=tf.train.get_global_step())

  train_hooks = None
  if not FLAGS.multi_gpu or opt._graph_state().is_the_last_tower:
    with open('../data/unsupervised_v4/word_counts_v4_pad.txt', 'r') as f:
      dic = list(f)
      dic = [i.split()[0] for i in dic]
      end_id = dic.index('</S>')
      dic.append('<unk>')
      dic = tf.convert_to_tensor(dic)
    noise = features['key'][0]
    m = tf.sequence_mask(features['len'][0], tf.shape(noise)[0])
    noise = tf.boolean_mask(noise, m)
    noise = tf.gather(dic, noise)
    sentence = crop_sentence(labels['sentence'][0], end_id)
    sentence = tf.gather(dic, sentence)
    pred = crop_sentence(predictions[0], end_id)
    pred = tf.gather(dic, pred)
    train_hooks = [tf.train.LoggingTensorHook(
      {'sentence': sentence, 'noise': noise, 'pred': pred, 'loss1':loss1,'loss2':loss2}, every_n_iter=100)]
    for variable in tf.trainable_variables():
      tf.summary.histogram(variable.op.name, variable)

  predictions = tf.boolean_mask(predictions, mask)
  metrics = {
    'acc': tf.metrics.accuracy(targets, predictions)
  }


  def init_fn(scaffold, session):
    inc_saver.restore(session, FLAGS.inc_ckpt)

  scaffold = tf.train.Scaffold(init_fn=init_fn)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    scaffold=scaffold,
    train_op=train_op,
    training_hooks=train_hooks,
    eval_metric_ops=metrics)

def batching_func(x, batch_size):
  return x.padded_batch(
    batch_size,
    padded_shapes=((
                    tf.TensorShape([299, 299, 3]),
                    tf.TensorShape([None]),
                    tf.TensorShape([])),
                   (tf.TensorShape([None]),
                    tf.TensorShape([]),
                    tf.TensorShape([None]),
                    tf.TensorShape([]))),
    drop_remainder=True)

def parse_sentence(serialized):
  """Parses a tensorflow.SequenceExample into an caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.

  Returns:
    key: The keywords in a sentence.
    num_key: The number of keywords.
    sentence: A description.
    sentence_length: The length of the description.
  """
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={},
    sequence_features={
      'key': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'sentence': tf.FixedLenSequenceFeature([], dtype=tf.int64),
    })
  key = tf.to_int32(sequence['key']) + 1 
  key = tf.random_shuffle(key)
  sentence = tf.to_int32(sequence['sentence']) +1 
  sentence_pad = tf.concat([sentence,[FLAGS.pad_id] * FLAGS.pad_length], axis=0)
  return key, tf.shape(key)[0], sentence_pad, tf.shape(sentence)[0]

def preprocess_image(encoded_image, classes):
  """Decodes an image."""
  image = tf.image.decode_jpeg(encoded_image, 3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize_images(image, [346, 346])
  image = tf.random_crop(image, [299, 299, 3])
  image = image * 2 - 1
  return image, classes, tf.shape(classes)[0]

def parse_image(serialized):
  """Parses a tensorflow.SequenceExample into an image and detected objects.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    classes: A 1-D int64 Tensor containing the detected objects.
    scores: A 1-D float32 Tensor containing the detection scores.
  """
  context, sequence = tf.parse_single_sequence_example(
    serialized,
    context_features={
      'image/data': tf.FixedLenFeature([], dtype=tf.string)
    },
    sequence_features={
      'classes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
    })

  encoded_image = context['image/data']
  classes = tf.to_int32(sequence['classes'])+1
  classes = classes[-5:]
  return encoded_image, classes

def input_fn(batch_size, subset='train'):
  sentence_ds = tf.data.TFRecordDataset('../data/unsupervised_v4/sentence_v4.tfrec')
  sentence_ds = sentence_ds.map(parse_sentence, num_parallel_calls=AUTOTUNE)
  sentence_ds = sentence_ds.filter(lambda k, lk, s, ls: tf.not_equal(lk, 0))
  if subset == 'train':
    sentence_ds = sentence_ds.apply(tf.contrib.data.shuffle_and_repeat(65536))
  
  image_ds = tf.data.TFRecordDataset('../data/unsupervised_v4/image_train_v4.tfrec')
  image_ds = image_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
  image_ds = image_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE) 
  image_ds = image_ds.filter(lambda im, k, lk: tf.not_equal(lk, 0))
  image_ds = image_ds.shuffle(8192).repeat()

  dataset = tf.data.Dataset.zip((image_ds, sentence_ds))
  dataset = batching_func(dataset, batch_size)
  dataset = dataset.prefetch(AUTOTUNE)
  iterator = dataset.make_one_shot_iterator()
  image, sentence = iterator.get_next()

  key, lk, sentence, ls = sentence
  im, classes, num = image

  return {'key': key, 'len': lk, 'im':im, 'classes': classes, 'num': num}, {'sentence': sentence ,'len': ls}

def main(_):
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '2'
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'

  if FLAGS.multi_gpu:
    validate_batch_size_for_multi_gpu(FLAGS.batch_size)
    model_function = tf.contrib.estimator.replicate_model_fn(
      model_fn,
      loss_reduction=tf.losses.Reduction.MEAN)
  else:
    model_function = model_fn

  sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
    inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
    gpu_options=tf.GPUOptions(allow_growth=True))

  run_config = tf.estimator.RunConfig(
    session_config=sess_config,
    save_checkpoints_steps=FLAGS.save_checkpoint_steps,
    save_summary_steps=FLAGS.save_summary_steps,
    keep_checkpoint_max=100)

  train_input_fn = functools.partial(input_fn, batch_size=FLAGS.batch_size)

  estimator = tf.estimator.Estimator(
    model_fn=model_function,
    model_dir=FLAGS.job_dir,
    config=run_config,
    params=FLAGS)

  estimator.train(train_input_fn, max_steps=FLAGS.max_steps)


if __name__ == '__main__':
  tf.app.run(main)
