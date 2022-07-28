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
import numpy as np


class LGTFB(tf.keras.layers.Layer):
  """
  """

  def __init__(self,
               kSize=1024,
               mode=modes.Modes.TRAINING,
               inference_batch_size=1,
               nBins=128,
               LS=3,
               nChan=1,
               en_kernel_size=9,
               EN=True,
               use_frame=False,
               name='LGTFB',
               **kwargs):
    super(LGTFB, self).__init__(**kwargs)

    self.ksize = kSize
    self.nbins = nBins  # number of hidden units in the first dense layer
    self.ls = LS  # use bias in the first dense layer
    self.use_frame=use_frame
    self.nchan = nChan  
    self.en_kernel_size = int(en_kernel_size)
    self.en = EN
    self.inference_batch_size = inference_batch_size
    self.mode = mode
    
    if self.use_frame:
      self.stride = self.ksize
    else:
      self.stride = 128
    self.pool_size = 2
    max_f = 7600.0
    min_f = 20.0
    mel_f  = np.linspace(min_f, 1127*np.log(1+max_f/700), self.nbins+1, True)   # num_chan+1 points from 0 to max_mel
    self.init_freq  = 700 * (np.exp(mel_f / 1127) - 1)
    self.level = np.log2(self.nbins).astype(int)
    tree_freq_init = self._freq2tf(self.init_freq, self.nbins, self.level)
    tree_freq_init = -np.log(1/tree_freq_init -1 + 1e-8) #inverse_sig
    tree_freq = 1.0/(1.0+np.exp(-tree_freq_init))
    [freq,nFreqDiff]  = self._tf2freq(tree_freq, self.level, self.ls)                
    freq           = np.reshape(freq, (1,nBins*LS,1))        
    nFreqDiff      = np.reshape(nFreqDiff, (1,nBins*LS,1)) * 2 
    # gamma parameter
    scale     = 1.0/(1.0+np.exp(-np.zeros((1,1,self.nchan), dtype=np.float32)))
    shape     = np.exp(np.zeros((1,1,self.nchan), dtype=np.float32))
    n      = np.cumsum(np.ones((kSize,nBins*LS,1),dtype=np.float32),0)	# [2048x128x1] 
    gamma_1  = np.power(n/self.ksize,shape-1)
    gamma_2  = np.exp(-np.pi*nFreqDiff*scale*n)
    self.gamma  = gamma_1 * gamma_2
    self.gamma  = self.gamma / np.mean(self.gamma,axis=0,keepdims=True)
    self.tone   = np.cos(np.pi*(freq*n))	
    self.kernel = self._custom_kernel(self.gamma,self.tone)
    
    #EN
    self.en_kernel = np.ones((self.en_kernel_size,self.en_kernel_size,self.ls,1),dtype=np.float32)


  def build(self, input_shape):
    super(LGTFB, self).build(input_shape)
    
    self.kernel_initializer = tf.keras.initializers.constant(self.kernel)
    self.conv2d = tf.keras.layers.Conv2D(filters=self.kernel.shape[3], kernel_size=(self.kernel.shape[0],self.kernel.shape[1]), strides=[self.stride,1], padding='VALID', kernel_initializer=self.kernel_initializer)
    #self.conv2d.set_weights([self.kernel])
    #self.conv2d = tf.nn.conv2d(input=inputs, filters=self.kernel, strides=[1,128,1,1], padding='VALID')
    #input_reshape = tf.reshape(input_shape, [input_shape.shape[0], input_shape.shape[1], self.nbins*self.ls, self.nchan])
    self.max_pool2d = tf.keras.layers.MaxPool2D(pool_size=[self.pool_size, self.ls], strides=[self.pool_size, self.ls], padding='VALID')
    #self.max_pool2d = tf.nn.max_pool2d(input=input_reshape, ksize=[1, 4, self.ls, 1], strides=[1, 4, self.ls, 1], padding='VALID')
    self.kernel_initializer2 = tf.keras.initializers.constant(self.en_kernel)
    self.mu_lsten = tf.keras.layers.DepthwiseConv2D((self.en_kernel_size, self.en_kernel_size),strides=(1,1),padding='SAME', kernel_initializer=self.kernel_initializer2)
    self.var_lsten = tf.keras.layers.DepthwiseConv2D((self.en_kernel_size, self.en_kernel_size),strides=(1,1),padding='SAME', kernel_initializer=self.kernel_initializer2)
    self.weight = tf.nn.softmax(tf.Variable(name='weight', shape=(2,3), initial_value=tf.constant([[-1,-1,1],[-1,-1,1]], dtype=tf.float32), dtype=tf.float32))
    
  def _custom_kernel(self, gamma, tone):
    kernel = gamma * tone		
    kernel = np.reshape(kernel,(self.ksize,1,1,self.nbins*self.ls*self.nchan))
    kernel /= np.sqrt(np.sum(kernel*kernel,axis=(0,1,2),keepdims=True)+1e-4)
    return kernel
    
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
    freq = np.zeros((1),dtype=np.float32)
    for lev in range(level):
      N = freq.shape[0]
      st = 2**lev-1
      ed = st*2 + 1
      freq = np.reshape(np.stack([freq,freq],1),(N*2))
      split = np.reshape(np.stack([np.log(tree_freq[st:ed]+1e-8),np.log(1-tree_freq[st:ed]+1e-8)],1),(N*2))
      freq = freq + split
    N = freq.shape[0]
    freq = np.stack([freq]*LS,1)
    split = np.log(np.array([[1.0/LS]*LS]).astype(np.float32))
    freq = np.reshape(freq + split, (N*LS))
    nFreqDiff = np.exp(freq)
    freq = np.cumsum(nFreqDiff)
    maxfreq = freq[-1]
    freq = freq / maxfreq
    nFreqDiff = nFreqDiff / maxfreq
    return [freq, nFreqDiff]
    

  def call(self, inputs, training=None):
    net = inputs
    if self.use_frame:
      net = tf.reshape(net, [net.shape[0],net.shape[1]*net.shape[2],1,1])
    else:
      net = tf.keras.backend.expand_dims(net, axis=2)
      net = tf.keras.backend.expand_dims(net, axis=3)
    net = self.conv2d(net)
    net = tf.math.log(tf.math.abs(net)+1)
    net = tf.reshape(net, [net.shape[0], net.shape[1], self.nbins*self.ls, self.nchan])
    net = self.max_pool2d(net)
    #EN
    if self.en:
      mu_SEN = tf.math.reduce_mean(net, axis=[1], keepdims=True)
      mu_TEN = tf.math.reduce_mean(net, axis=[2], keepdims=True)
      mu_LSTEN = self.mu_lsten(net)/(self.en_kernel_size*self.en_kernel_size)
      mu_ws = self.weight[0,0]*mu_SEN + self.weight[0,1]*mu_TEN + self.weight[0,2]*mu_LSTEN
      
      net_zm = net - mu_ws
      net_zm2 = tf.math.square(net_zm)
      
      var_SEN = tf.math.reduce_mean(net_zm2, axis=[1], keepdims=True)
      var_TEN = tf.math.reduce_mean(net_zm2, axis=[2], keepdims=True)
      var_LSTEN = self.var_lsten(net_zm2)/(self.en_kernel_size*self.en_kernel_size)
      var_ws = self.weight[1,0]*var_SEN + self.weight[1,1]*var_TEN + self.weight[1,2]*var_LSTEN
            
      net = tf.math.divide_no_nan(net_zm, tf.math.sqrt(var_ws+1))

    net = tf.reshape(net, [net.shape[0], net.shape[1], self.nbins*self.nchan])
    return net

  def get_config(self):
    config = {
        'kSize': self.ksize,
        'nBins': self.nbins,
        'LS': self.ls,
        'nChan': self.nchan,
        'en_kernel_size': self.en_kernel_size,
        'EN': self.en,
        'use_frame': self.use_frame,
        'inference_batch_size': self.inference_batch_size,
        'mode': self.mode,
    }
    base_config = super(LGTFB, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
