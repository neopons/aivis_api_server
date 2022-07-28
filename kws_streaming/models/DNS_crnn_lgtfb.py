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

"""Conv and RNN based model."""
from kws_streaming.layers import gru
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils
from kws_streaming.layers import dtln
from kws_streaming.layers import lgtfb
from tensorflow.python.ops import gen_audio_ops as audio_ops

def model_parameters(parser_nn):
  """CRNN model parameters."""
  parser_nn.add_argument(
      '--cnn_filters',
      type=str,
      default='16,16',
      help='Number of output filters in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn_kernel_size',
      type=str,
      default='(3,3),(5,3)',
      help='Heights and widths of the 2D convolution window',
  )
  parser_nn.add_argument(
      '--cnn_act',
      type=str,
      default="'relu','relu'",
      help='Activation function in the convolution layers',
  )
  parser_nn.add_argument(
      '--cnn_dilation_rate',
      type=str,
      default='(1,1),(1,1)',
      help='Dilation rate to use for dilated convolutions',
  )
  parser_nn.add_argument(
      '--cnn_strides',
      type=str,
      default='(1,1),(1,1)',
      help='Strides of the convolution layers along the height and width',
  )
  parser_nn.add_argument(
      '--gru_units',
      type=str,
      default='256',
      help='Output space dimensionality of gru layer',
  )
  parser_nn.add_argument(
      '--return_sequences',
      type=str,
      default='0',
      help='Whether to return the last output in the output sequence,'
      'or the full sequence',
  )
  parser_nn.add_argument(
      '--stateful',
      type=int,
      default='1',
      help='If True, the last state for each sample at index i'
      'in a batch will be used as initial state for the sample '
      'of index i in the following batch',
  )
  parser_nn.add_argument(
      '--dropout1',
      type=float,
      default=0.1,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--units1',
      type=str,
      default='128,256',
      help='Number of units in the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--act1',
      type=str,
      default="'linear','relu'",
      help='Activation function of the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--dtln_units',
      type=str,
      default='128',
      help='Activation function of the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--dtln_num_layer',
      type=str,
      default='2',
      help='Activation function of the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--dtln_blockLen',
      type=str,
      default='512',
      help='Activation function of the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--dtln_block_shift',
      type=str,
      default='128',
      help='Activation function of the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--dtln_encoder_size',
      type=str,
      default='256',
      help='Activation function of the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--dtln_dropout',
      type=float,
      default=0.25,
      help='Activation function of the last set of hidden layers',
  )
  parser_nn.add_argument(
      '--dtln_activation',
      type=str,
      default='sigmoid',
      help='Activation function of the last set of hidden layers',
  )


def model(flags):
  """Deep Noise Supression Convolutional recurrent neural network (CRNN) model.

  It is based on paper
  Convolutional Recurrent Neural Networks for Small-Footprint Keyword Spotting
  https://arxiv.org/pdf/1703.05390.pdf
  Represented as sequence of Conv, RNN/GRU, FC layers.
  Model topology is similar with "Hello Edge: Keyword Spotting on
  Microcontrollers" https://arxiv.org/pdf/1711.07128.pdf
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """
  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  
  flags.feature_type = 'time'
  net = speech_features.SpeechFeatures(speech_features.SpeechFeatures.get_params(flags))(input_audio)
  
  net_mag,net_angle = tf.keras.layers.Lambda(stftLayer)((net, flags.dtln_blockLen, flags.dtln_block_shift))
  
  mask_1 = dtln.DTLN(units=flags.dtln_units, num_layer = flags.dtln_num_layer, dropout = flags.dtln_dropout, mask_size = (int(flags.dtln_blockLen)//2+1), activation = flags.dtln_activation)(net_mag)

  net_mag = tf.keras.layers.Multiply()([net_mag, mask_1])
  net = tf.keras.layers.Lambda(ifftLayer)([net_mag, net_angle])
  
  net = tf.keras.layers.Conv1D(int(flags.dtln_encoder_size),1,strides=1,use_bias=False)(net)
  # normalize the input to the separation kernel
  net = InstantLayerNormalization()(net)
  # predict mask based on the normalized feature frames
  mask_2 = dtln.DTLN(units=flags.dtln_units, num_layer = flags.dtln_num_layer, dropout = flags.dtln_dropout, mask_size = flags.dtln_encoder_size, activation = flags.dtln_activation)(net)
  # multiply encoded frames with the mask
  net = tf.keras.layers.Multiply()([net, mask_2]) 
  # decode the frames back to time domain
  net = tf.keras.layers.Conv1D(int(flags.dtln_blockLen), 1, padding='causal',use_bias=False)(net)
  # create waveform with overlap and add procedure
  #net = tf.keras.layers.Lambda(overlapAddLayer)((net, flags.dtln_block_shift))
  net = lgtfb.LGTFB(kSize=int(net.shape[-1]), use_frame=True)(net)
  #flags.feature_type = 'LMFB_tf_from_frames'
  #net = speech_features.SpeechFeatures(speech_features.SpeechFeatures.get_params(flags))(net)
  # expand dims for the next layer 2d conv
  net = tf.keras.backend.expand_dims(net)
  for filters, kernel_size, activation, dilation_rate, strides in zip(
      utils.parse(flags.cnn_filters), utils.parse(flags.cnn_kernel_size),
      utils.parse(flags.cnn_act), utils.parse(flags.cnn_dilation_rate),
      utils.parse(flags.cnn_strides)):
    net = stream.Stream(
        cell=tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            dilation_rate=dilation_rate,
            strides=strides))(
                net)

  shape = net.shape
  # input net dimension: [batch, time, feature, channels]
  # reshape dimension: [batch, time, feature * channels]
  # so that GRU/RNN can process it
  net = tf.keras.layers.Reshape((-1, shape[2] * shape[3]))(net)

  for units, return_sequences in zip(
      utils.parse(flags.gru_units), utils.parse(flags.return_sequences)):
    net = gru.GRU(
        units=units, return_sequences=return_sequences,
        stateful=flags.stateful)(
            net)

  net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
  net = tf.keras.layers.Dropout(rate=flags.dropout1)(net)

  for units, activation in zip(
      utils.parse(flags.units1), utils.parse(flags.act1)):
    net = tf.keras.layers.Dense(units=units, activation=activation)(net)

  net = tf.keras.layers.Dense(units=flags.label_count)(net)
  if flags.return_softmax:
    net = tf.keras.layers.Activation('softmax')(net)
  return tf.keras.Model(input_audio, net)
  
def stftLayer(inputs):
    '''
    Method for an STFT helper layer used with a Lambda layer. The layer
    calculates the STFT on the last dimension and returns the magnitude and
    phase of the STFT.
    '''
    x = inputs[0]
    dtln_blockLen = int(inputs[1])
    dtln_block_shift = int(inputs[2])
    # creating frames from the continuous waveform
    frames = tf.signal.frame(x, dtln_blockLen, dtln_block_shift)
    # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
    stft_dat = tf.signal.rfft(frames)
    # calculating magnitude and phase from the complex signal
    mag = tf.abs(stft_dat)
    phase = tf.math.angle(stft_dat)
    # returning magnitude and phase as list
    return [mag, phase]
            
def ifftLayer(x):
    '''
    Method for an inverse FFT layer used with an Lambda layer. This layer
    calculates time domain frames from magnitude and phase information. 
    As input x a list with [mag,phase] is required.
    '''
        
    # calculating the complex representation
    s1_stft = (tf.cast(x[0], tf.complex64) * 
                tf.exp( (1j * tf.cast(x[1], tf.complex64))))
    # returning the time domain frames
    return tf.signal.irfft(s1_stft)  

def overlapAddLayer(inputs):
    '''
    Method for an overlap and add helper layer used with a Lambda layer.
    This layer reconstructs the waveform from a framed signal.
    '''

    x = inputs[0]
    dtln_block_shift = int(inputs[1])
    # calculating and returning the reconstructed waveform
    return tf.signal.overlap_and_add(x, dtln_block_shift)
    
class InstantLayerNormalization(tf.keras.layers.Layer):
    '''
    Class implementing instant layer normalization. It can also be called 
    channel-wise layer normalization and was proposed by 
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2) 
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7 
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                             initializer='ones',
                             trainable=True,
                             name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                             initializer='zeros',
                             trainable=True,
                             name='beta')
 

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), 
                                       axis=[-1], keepdims=True)
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently 
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs
        

