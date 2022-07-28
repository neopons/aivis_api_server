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

"""Supported models."""
import kws_streaming.models.crnn as crnn
import kws_streaming.models.ds_cnn as ds_cnn
import kws_streaming.models.kws_transformer as kws_transformer
import kws_streaming.models.DNS_crnn as DNS_crnn
import kws_streaming.models.crnn_lgtfb as crnn_lgtfb
import kws_streaming.models.DNS_crnn_lgtfb as DNS_crnn_lgtfb

# dict with supported models
MODELS = {
    'ds_cnn': ds_cnn.model,
    'crnn': crnn.model,
    'kws_transformer': kws_transformer.model,
    'DNS_crnn': DNS_crnn.model,
    'crnn_lgtfb': crnn_lgtfb.model,
    'DNS_crnn_lgtfb': DNS_crnn_lgtfb.model
}
