# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train utility functions, based on tensorflow/examples/speech_commands.

  It consists of several steps:
  1. Creates model.
  2. Reads data
  3. Trains model
  4. Select the best model and evaluates it
"""

import os.path
import pprint
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import kws_streaming.data.input_data as input_data
from kws_streaming.models import models
from kws_streaming.models import utils
from transformers import AdamWeightDecay
import math
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features

def transfer_train(flags):
  """Model training."""

  flags.training = True

  # Set the verbosity based on flags (default is INFO, so we see all messages)
  logging.set_verbosity(flags.verbosity)

  # Start a new TensorFlow session.
  tf.reset_default_graph()

  # allow_soft_placement solves issue with
  # "No device assignments were active during op"
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  tf.keras.backend.set_session(sess)

  audio_processor = input_data.AudioProcessor(flags)

  time_shift_samples = int((flags.time_shift_ms * flags.sample_rate) / 1000)

  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, flags.how_many_training_steps.split(',')))
  transfer_steps_list = list(map(int, flags.how_many_transfer_steps.split(',')))
  learning_rates_list = list(map(float, flags.learning_rate.split(',')))
  if len(transfer_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))
  logging.info(flags)
  logging.info(flags.model_name)
  lgtfb_model = models.MODELS['dns_lgtfb'](flags)

  #print(kws_model.layers[3].get_weights())
  input_audio = tf.keras.layers.Input(shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING), batch_size=flags.batch_size)
  net = lgtfb_model(input_audio)
  #net = tf.keras.layers.MaxPool2D(pool_size=(9,flags.lgtfb_nChan),strides=(3,1),padding='valid',data_format='channels_first')(net)
  if flags.lgtfb:
    net = tf.keras.layers.Conv2D(filters=net.shape[1],kernel_size=(9,flags.lgtfb_nChan),strides=(3,1),padding='valid',data_format='channels_first')(net)
    net = tf.squeeze(net)
  flags.feature_type = 'only_specaug'
  net = speech_features.SpeechFeatures(speech_features.SpeechFeatures.get_params(flags))(net)
  flags.lgtfb = False
  flags.dns = False
  flags.feature_type = 'mfcc_tf'
  kws_model = models.MODELS[flags.model_name](flags)
  #print(kws_model.layers[3].get_weights())
  kws_model.load_weights(os.path.join(flags.transfer_dir,'best_weights')).expect_partial()  #print(net.shape)

  input_sub = tf.keras.layers.Input(shape=net.shape[1:], batch_size=flags.batch_size)
  net_sub = input_sub
  for idx in range(2,len(kws_model.layers)):
    print(net_sub.shape)
    net_sub = kws_model.get_layer(index=idx)(net_sub, training=flags.training)
  backend_kws_model = tf.keras.Model(input_sub, net_sub)
  net = backend_kws_model(net)
  model = tf.keras.Model(input_audio, net)
  model.load_weights(os.path.join(flags.train_dir,'best_weights')).expect_partial()
  #print(backend_kws_model.layers[2].get_weights())
  #print(model.layers)
  #logging.info(model.summary())
  # fix for InvalidArgumentError:
  # Node 'Adam/gradients/gradients/gru/cell/while_grad/gru/cell/while_grad':
  # Connecting to invalid output 51 of source node
  # gru/cell/while which has 51 outputs.
  tf.compat.v1.experimental.output_all_intermediates(True)

  # save model summary
  utils.save_model_summary(model, flags.train_dir)

  # save model and data flags
  with open(os.path.join(flags.train_dir, 'flags.txt'), 'wt') as f:
    pprint.pprint(flags, stream=f)

