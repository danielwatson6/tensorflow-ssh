import sys

import numpy as np
import tensorflow as tf


try:
  USE_GPU = bool(sys.argv[1])
except IndexError:
  USE_GPU = False


### HYPERPARAMETERS

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
TIME_STEPS = 64
MODEL_SIZE = 256
NUM_GRU = 2


### READ DATA

with open('bible.txt') as f:
  DATA = f.read()

ID_TO_CHAR = list(set(DATA))
CHAR_TO_ID = {k: v for v, k in enumerate(ID_TO_CHAR)}

data = [CHAR_TO_ID[ch] for ch in DATA]
# We set 80% of the data for training and the rest for validation
n = int(len(data) * .8)
TRAIN_DATA, VALID_DATA = data[:n], data[n:]
del data, n

def get_batch(dataset):
  """Returns `BATCH_SIZE` arrays of textual sequences, each with `TIME_STEPS` + 1
     characters. We add an extra character to compensate for the labels."""
  batch = []
  for _ in range(BATCH_SIZE):
    n = np.random.randint(len(dataset) - TIME_STEPS - 1)
    batch.append(dataset[n:n+TIME_STEPS+1])
  return batch


### TOY LANGUAGE MODEL

def rnn_cell():
  """Return an instance of a new GRU cell, maybe assigned to run in the GPU."""
  cell = tf.contrib.rnn.GRUBlockCell(MODEL_SIZE)
  if USE_GPU:
    cell = tf.nn.rnn_cell.DeviceWrapper(cell, '/gpu:0')
  return cell

global_step = tf.Variable(0, trainable=False, name='global_step')

inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, TIME_STEPS))
labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE, TIME_STEPS))

E = tf.get_variable(
  'embeddings',
  shape=(len(ID_TO_CHAR), MODEL_SIZE),
  initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))

embeddings = tf.nn.embedding_lookup(E, inputs)

cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(NUM_GRU)])
rnn_out, _ = tf.nn.dynamic_rnn(cell, embeddings, dtype=tf.float32)

if USE_GPU:
  device = '/gpu:0'
else:
  device = '/cpu:0'
with tf.device(device):
  logits = tf.layers.dense(rnn_out, len(ID_TO_CHAR))

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
  labels=labels, logits=logits)

# Clip gradients by maximum norm of 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)

output = tf.argmax(logits, axis=2)
perplexity = tf.exp(tf.reduce_mean(loss))
metrics = tf.summary.scalar('perplexity', perplexity)


### RUN EXPERIMENT

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if USE_GPU:
  summary_dir = 'gpu'
else:
  summary_dir = 'cpu'
train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
valid_writer = tf.summary.FileWriter(summary_dir + '/valid', sess.graph)

def make_fd(batch):
  """Given a batch, get the inputs and labels ready to feed to the graph."""
  batch_inputs = [seq[:-1] for seq in batch]
  batch_labels = [seq[1:] for seq in batch]
  return {inputs: batch_inputs, labels: batch_labels}

while True:
  step = sess.run(global_step)
  train_batch = get_batch(TRAIN_DATA)
  valid_batch = get_batch(VALID_DATA)
  _, train_ppx, train_met = sess.run(
    [train_op, perplexity, metrics], feed_dict=make_fd(train_batch))
  valid_ppx, valid_out, valid_met = sess.run(
    [perplexity, output, metrics], feed_dict=make_fd(valid_batch))
  print("Step {0} (train_ppx={1}, valid_ppx={2})".format(
    step, train_ppx, valid_ppx))
  print("Sample validation output:")
  print(''.join([ID_TO_CHAR[i] for i in valid_out[0]]))
  train_writer.add_summary(train_met, step)
  valid_writer.add_summary(valid_met, step)

