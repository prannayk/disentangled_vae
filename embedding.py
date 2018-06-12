# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imsave

from model import VAE
from data_manager import DataManager

tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("gamma", 100.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 20.0,
        "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
        "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")

flags = tf.app.flags.FLAGS

def main(argv):
  print("Loading data")
  manager = DataManager()
  manager.load()
  print("Data load complete")
  sess = tf.Session()

  model = VAE(gamma=flags.gamma,
          capacity_limit=flags.capacity_limit,
          capacity_change_duration=flags.capacity_change_duration,
          learning_rate=flags.learning_rate)
  n = 10
  embedding_var = tf.Variable([n*flags.batch_size, 10],name="embedding")
  embedding_variables = [tf.Variable([flags.batch_size,10]) for i in range(n) ]
  place = tf.placeholder(tf.float32, shape=[flags.batch_size, 4096])
  set_embedding_slices = [tf.assign(embedding_var)]
  set_embedding
  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess)

