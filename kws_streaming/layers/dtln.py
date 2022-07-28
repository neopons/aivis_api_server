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


class DTLN(tf.keras.layers.Layer):
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
               units=64,
               mode=modes.Modes.TRAINING,
               inference_batch_size=1,
               return_sequences=True,
               use_peepholes=False,
               num_proj=128,
               unroll=False,
               stateful=False,
               name='DTLN',
               num_layer = 2,
               dropout = 0.25,
               mask_size = 257,
               activation = 'sigmoid',
               **kwargs):
    super(DTLN, self).__init__(**kwargs)

    self.mode = mode
    self.inference_batch_size = inference_batch_size
    self.units = int(units)
    self.return_sequences = return_sequences
    self.num_proj = num_proj
    self.use_peepholes = use_peepholes
    self.stateful = int(stateful)
    self.num_layer = int(num_layer)
    self.dropout = float(dropout)
    self.mask_size = int(mask_size)
    self.activation = activation

    if mode != modes.Modes.TRAINING:  # in any inference mode
      # let's unroll lstm, so there is no symbolic loops / control flow
      unroll = True

    self.unroll = unroll
    if self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      if use_peepholes:
        raise ValueError ('support later')
      else:
        for idx in range(self.num_layer):
            globals()[f'self.lstm_{idx}'] = tf.keras.layers.LSTM(
                units=self.units,
                return_sequences=self.return_sequences,
                unroll=self.unroll,
                stateful=self.stateful)
        for idx in range(self.num_layer-1):
            globals()[f'self.dropout_{idx}'] = tf.keras.layers.Dropout(rate=self.dropout)
        self.dense = tf.keras.layers.Dense(self.mask_size)
        self.activation = tf.keras.layers.Activation(self.activation)
    if self.mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE or self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      raise ValueError(f'{self.mode} : support later')

  def call(self, inputs):
    if inputs.shape.rank != 3:  # [batch, time, feature]
      raise ValueError('inputs.shape.rank:%d must be 3' % inputs.shape.rank)

    if self.mode in (modes.Modes.TRAINING, modes.Modes.NON_STREAM_INFERENCE):
      # run non streamable training or non streamable inference
      # on input [batch, time, features], returns [batch, time, units]
      return self._non_streaming(inputs)
    else:
      raise ValueError(f'support later `{self.mode}`.')

  def get_config(self):
    config = {
        'mode': self.mode,
        'inference_batch_size': self.inference_batch_size,
        'units': self.units,
        'return_sequences': self.return_sequences,
        'unroll': self.unroll,
        'num_proj': self.num_proj,
        'use_peepholes': self.use_peepholes,
        'stateful': self.stateful,
    }
    base_config = super(LSTM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_input_state(self):
    # input state is used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.input_state1, self.input_state2]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def get_output_state(self):
    # output state is used only for STREAM_EXTERNAL_STATE_INFERENCE mode
    if self.mode == modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE:
      return [self.output_state1, self.output_state2]
    else:
      raise ValueError('Expected the layer to be in external streaming mode, '
                       f'not `{self.mode}`.')

  def _non_streaming(self, inputs):
    # inputs [batch, time, feature]
    x = inputs
    for idx in range(self.num_layer):
        x = globals()[f'self.lstm_{idx}'](x)
        # using dropout between the LSTM layer for regularization 
        if idx<(self.num_layer-1):
            x = globals()[f'self.dropout_{idx}'](x)
        # creating the mask with a Dense and an Activation layer
    mask = self.dense(x)
    output = self.activation(mask)

    if not self.return_sequences:
      # if we do not return sequence the output will be [batch, units]
      # for consistency make it [batch, 1, units]
      output = tf.keras.backend.expand_dims(output, axis=1)
    return output
