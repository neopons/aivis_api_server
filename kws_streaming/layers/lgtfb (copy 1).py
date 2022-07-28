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

"""LSTM layer."""

from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf
from kws_streaming.layers.compat import tf1
import numpy as np
import tensorflow as tf2


class LGTFB(tf.keras.layers.Layer):
  """LSTM with support of streaming inference with internal/external state.

  In training mode we use LSTM.
  It receives input data [batch, time, feature] and
  returns [batch, time, units] if return_sequences==True or
  returns [batch, 1, units] if return_sequences==False

  In inference mode we use LSTMCell
  In streaming mode with internal state
  it receives input data [batch, 1, feature]
  In streaming mode with internal state it returns: [batch, 1, units]

  In streaming mode with external state it receives input data with states:
    [batch, 1, feature] + state1[batch, units] + state2[batch, units]
  In streaming mode with external state it returns:
    (output[batch, 1, units], state1[batch, units], state2[batch, units])
  We use layer and parameter description from:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
  https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

  Attributes:
    units: dimensionality of the output space.
    mode: Training or inference modes: non streaming, streaming.
    inference_batch_size: batch size for inference mode
    return_sequences: Whether to return the last output. in the output sequence,
      or the full sequence.
    use_peepholes: True to enable diagonal/peephole connections
    num_proj: The output dimensionality for the projection matrices. If None, no
      projection is performed. It will be used only if use_peepholes is True.
    unroll:  If True, the network will be unrolled, else a symbolic loop will be
      used. For any inference mode it will be set True inside.
    stateful: If True, the last state for each sample at index i in a batch will
      be used as initial state for the sample of index i in the following batch.
      If model will be in streaming mode then it is better to train model with
      stateful=True This flag is about stateful training and applied during
      training only.
  """

  def __init__(self,
               kSize=2048,
               mode=modes.Modes.TRAINING,
               inference_batch_size=1,
               nBins=128,
               LS=3,
               nChan=3,
               name='LGTFB',
               **kwargs):
    super(LGTFB, self).__init__(**kwargs)

    self.ksize = kSize
    self.nbins = nBins  # number of hidden units in the first dense layer
    self.ls = LS  # use bias in the first dense layer

    self.nchan = nChan  
    self.inference_batch_size = inference_batch_size
    self.mode = mode
    
    max_f = 7600.0
    min_f = 20.0
    mel_f  = np.linspace(min_f, 1127*np.log(1+max_f/700), self.nbins+1, True)   # num_chan+1 points from 0 to max_mel
    self.init_freq  = 700 * (np.exp(mel_f / 1127) - 1)
    self.level = np.log2(self.nbins).astype(int)
    tree_freq_init = self._freq2tf(self.init_freq, self.nbins, self.level)
    tree_freq_init = -np.log(1/tree_freq_init -1 + 1e-8) #inverse_sig
    tree_freq      = tf.math.sigmoid(tf.Variable(name='freq', shape=[nBins-1], initial_value=tree_freq_init, dtype=tf.float32))
    [freq,nFreqDiff]  = self._tf2freq(tree_freq, self.level, self.ls)                
    freq           = tf.reshape(freq, [1,nBins*LS,1])        
    nFreqDiff      = tf.reshape(nFreqDiff, [1,nBins*LS,1]) * 2 
    # gamma parameter
    scale     = tf.math.sigmoid(tf.Variable(name='scale', shape=[1,1,nChan], initial_value=tf.zeros(shape=[1,1,nChan], dtype=tf.float32)))
    shape     = tf.math.exp(tf.Variable(name='shape', shape=[1,1,nChan], initial_value=tf.zeros(shape=[1,1,nChan], dtype=tf.float32)))
    n      = tf.math.cumsum(tf.ones(shape=[kSize,nBins*LS,1],dtype=tf.float32),0)	# [2048x128x1] 
    gamma_1  = tf.math.pow(n/self.ksize,shape-1)
    gamma_2  = tf.math.exp(-np.pi*nFreqDiff*scale*n)
    self.gamma  = gamma_1 * gamma_2
    self.gamma  = self.gamma / tf.reduce_mean(input_tensor=self.gamma,axis=0,keepdims=True)
    self.tone   = tf.math.cos(np.pi*(freq*n))	
    self.kernel = self._custom_kernel(self.gamma,self.tone)
    #self.kernel = tf2.constant(self.kernel)


  def build(self, input_shape):
    super(LGTFB, self).build(input_shape)
    
    self.kernel_initializer = tf.keras.initializers.constant((self.kernel).numpy())
    self.conv2d = tf.keras.layers.Conv2D(filters=self.kernel.shape[3], kernel_size=(self.kernel.shape[0],self.kernel.shape[1]), strides=[128,1], padding='VALID', kernel_initializer=self.kernel_initializer)
    #self.conv2d.set_weights([self.kernel])
    #self.conv2d = tf.nn.conv2d(input=inputs, filters=self.kernel, strides=[1,128,1,1], padding='VALID')
    #input_reshape = tf.reshape(input_shape, [input_shape.shape[0], input_shape.shape[1], self.nbins*self.ls, self.nchan])
    self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size=[4, self.ls], strides=[4, self.ls], padding='VALID')
    #self.max_pool2d = tf.nn.max_pool2d(input=input_reshape, ksize=[1, 4, self.ls, 1], strides=[1, 4, self.ls, 1], padding='VALID')
    
  def _custom_kernel(self, gamma, tone):
    kernel = gamma * tone		
    kernel = tf.reshape(kernel,[self.ksize,1,1,self.nbins*self.ls*self.nchan])
    kernel /= tf.math.sqrt(tf.reduce_sum(input_tensor=kernel*kernel,axis=[0,1,2],keepdims=True)+1e-4)
    return kernel
    
  def _custom_weigths(self, shape, dtype=None):
    print(self.kernel)
    return tf.keras.backend.Variable(value=self.kernel, dtype=dtype)
    
  def _freq2tf(self, freq, nBins, level):
    freq = freq / freq[-1]                         # in [0~1]
    centBin = int(nBins/2)
    tree_freq = freq[centBin:centBin+1]
    for lev in range(1,level):
      for n in range(2**lev):
        step = int(nBins / (2**lev))
        st = int(step*n)
        ed = int(step*(n+1))
        cr = int((st+ed)/2)
        tree_freq = np.concatenate((tree_freq, [(freq[cr]-freq[st])/(freq[ed]-freq[st])]))
    return tree_freq

  def _tf2freq(self, tree_freq, level, LS):
    freq = tf.zeros(shape=[1],dtype=tf.float32)
    for lev in range(level):
      N = freq.get_shape().as_list()[0]
      st = 2**lev-1
      ed = st*2 + 1
      freq = tf.reshape(tf.stack([freq,freq],1),shape=[N*2])
      split = tf.reshape(tf.stack([tf.math.log(tree_freq[st:ed]+1e-8),tf.math.log(1-tree_freq[st:ed]+1e-8)],1),shape=[N*2])
      freq = freq + split
    N = freq.get_shape().as_list()[0]
    freq = tf.stack([freq]*LS,1)
    split = tf.math.log(tf.constant([[1.0/LS]*LS],dtype=tf.float32))
    freq = tf.reshape(freq + split, shape=[N*LS])
    nFreqDiff = tf.math.exp(freq)
    freq = tf.math.cumsum(nFreqDiff)
    maxfreq = freq[-1]
    freq = freq / maxfreq
    nFreqDiff = nFreqDiff / maxfreq
    return [freq, nFreqDiff]
    

  def call(self, inputs, training=None):
    net = inputs
    net = tf.keras.backend.expand_dims(net, axis=2)
    net = tf.keras.backend.expand_dims(net, axis=3)
    net = self.conv2d(net)
    net = tf.math.log(tf.math.abs(net)+1)
    print(net.shape)
    net = tf.reshape(net, [net.shape[0], net.shape[1], self.nbins*self.ls, self.nchan])
    print(net.shape)
    net = self.max_pool2d(net)
    print(net.shape)

    return net

  def get_config(self):
    config = {
        'kSize': self.ksize,
        'nBins': self.nbins,
        'LS': self.ls,
        'nChan': self.nchan,
        'inference_batch_size': self.inference_batch_size,
        'mode': self.mode,
    }
    base_config = super(LGTFB, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
