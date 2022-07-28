# coding=utf-8
from encodings import normalize_encoding
from ipaddress import summarize_address_range
from lib2to3.pgen2.token import AMPEREQUAL
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import sys
import webrtcvad
import numpy as np
import json
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import tensorflow.lite as tflite
from kws_streaming.models import models
import math

logging.basicConfig(
    stream=sys.stdout,
    format="%(levelname)-8s %(asctime)-15s %(name)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def vad_function(sample_rate, frame_duration_ms, vad_percent, vad, audio):
    audio_pcm = float2pcm(audio)
    frames = frame_generator(frame_duration_ms, audio_pcm, sample_rate)
    frames = list(frames)
    total = len(frames)
    vad_frame_num = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            vad_frame_num += 1
    output_percent = vad_frame_num / (total * 1.0)
    if output_percent > vad_percent:
        vad_result = True
    else:
        vad_result = False
    return vad_result


def vad_function2(sample_rate, vad, audio, frame_duration_ms=30):
    audio_pcm = float2pcm(audio)
    frames = frame_generator(frame_duration_ms, audio_pcm, sample_rate)
    frames = list(frames)
    total = len(frames)
    vad_frame_num = 0
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            vad_frame_num += 1
    output_percent = vad_frame_num / (total * 1.0)
    return output_percent


def energy_function(sample_rate, audio, start_band=300, end_band=7200):
    amplitude = np.abs(np.fft.fft(audio))
    amplitude = amplitude[1:]
    energy = amplitude ** 2

    frequency = np.fft.fftfreq(len(audio), 1.0 / sample_rate)
    frequency = frequency[1:]
    normalize_energy = {}
    for (i, freq) in enumerate(frequency):
        if abs(freq) not in normalize_energy:
            normalize_energy[abs(freq)] = energy[i] * 2

    sum_energy = 0
    for f in normalize_energy.keys():
        if start_band < f < end_band:
            sum_energy += normalize_energy[f]
    # full_energy = sum(normalize_energy.values())
    # voice_rate = sum_energy/(full_energy*1.0)
    return sum_energy


def read_labels(filename):
    # The labels file can be made something like this.
    f = open(filename, "r")
    lines = f.readlines()
    return [l.rstrip() for l in lines]


def load_vad_model(vad_level):
    vad_model = webrtcvad.Vad(int(vad_level))
    return vad_model


def load_kws_model(model_path):
    with tf.compat.v1.gfile.Open(os.path.join(model_path, 'flags.json'), 'r') as fd:
        flags_json = json.load(fd)

    class DictStruct(object):
        def __init__(self, **entries):
            self.__dict__.update(entries)

    flags = DictStruct(**flags_json)

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(sess)
    tf1.keras.backend.set_learning_phase(0)
    inference_batch_size = 1
    flags.batch_size = inference_batch_size

    kws_model = models.MODELS[flags.model_name](flags)
    weights_name = 'best_weights'
    kws_model.load_weights(os.path.join(model_path, weights_name)).expect_partial()

    print("kws model loaded")
    return kws_model


def load_aec_model(model_path, model_type):
    if model_type == "tf":
        aec_model = tf.saved_model.load(model_path)
        aec_infer = aec_model.signatures["serving_default"]
        aec_infer._num_positional_args = 2
    if model_type == "tflite":
        interpreter_1 = tflite.Interpreter(model_path=model_path + "_1.tflite")
        interpreter_1.allocate_tensors()
        interpreter_2 = tflite.Interpreter(model_path=model_path + "_2.tflite")
        interpreter_2.allocate_tensors()
        aec_infer = [interpreter_1, interpreter_2]
        # get details from interpreters
        input_details_1 = interpreter_1.get_input_details()
        input_details_2 = interpreter_2.get_input_details()
        # preallocate states for lstms
        states_1 = np.zeros(input_details_1[1]["shape"]).astype("float32")
        states_2 = np.zeros(input_details_2[1]["shape"]).astype("float32")
        states_buffer = [states_1, states_2]

    print("aec model loaded")
    return aec_infer, states_buffer


def inference_aec(infer, out_buffer, states_buffer, input_audio, lpb_audio, len_audio, model_type):
    out_buffer = np.squeeze(out_buffer)
    audio = np.squeeze(input_audio)
    lpb = np.squeeze(lpb_audio)
    # set block len and block shift
    block_len = 512
    block_shift = 128

    # audio = np.concatenate((cache_input_audio[-(block_len - block_shift):], audio))
    audio = audio[-(len_audio + block_len - block_shift):]
    lpb = lpb[-(len_audio + block_len - block_shift):]
    if model_type == 'tflite':
        interpreter_1 = infer[0]
        interpreter_2 = infer[1]
        # get details from interpreters
        input_details_1 = interpreter_1.get_input_details()
        output_details_1 = interpreter_1.get_output_details()
        input_details_2 = interpreter_2.get_input_details()
        output_details_2 = interpreter_2.get_output_details()
        # preallocate states for lstms
        states_1 = states_buffer[0]
        states_2 = states_buffer[1]
    # preallocate out file
    out_file = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len)).astype("float32")
    in_buffer_lpb = np.zeros((block_len)).astype("float32")
    # out_buffer = cache_output_audio[:].astype("float32")
    # calculate number of frames
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    # iterate over the number of frames
    for idx in range(num_blocks):
        # shift values and write to buffer of the input audio
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[
                                   idx * block_shift: (idx * block_shift) + block_shift
                                   ]
        # shift values and write to buffer of the loopback audio
        in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
        in_buffer_lpb[-block_shift:] = lpb[
                                       idx * block_shift: (idx * block_shift) + block_shift
                                       ]

        if model_type == "tf":
            in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
            in_block_lpb = np.expand_dims(in_buffer_lpb, axis=0).astype('float32')

            out_block = infer(tf.constant(in_block_lpb), tf.constant(in_block))['conv1d_2']
        else:
            # calculate fft of input block
            in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype("complex64")
            # create magnitude
            in_mag = np.abs(in_block_fft)
            in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
            # calculate log pow of lpb
            lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype("complex64")
            lpb_mag = np.abs(lpb_block_fft)
            lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
            # set tensors to the first model
            interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
            interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
            interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
            # run calculation
            interpreter_1.invoke()
            # # get the output of the first block
            out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
            states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])
            # apply mask and calculate the ifft
            estimated_block = np.fft.irfft(in_block_fft * out_mask)
            # reshape the time domain frames
            estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
            in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")
            # set tensors to the second block
            interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
            interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
            interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
            # run calculation
            interpreter_2.invoke()
            # get output tensors
            out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
            states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])

        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        # write block to output file
        out_file[idx * block_shift: (idx * block_shift) + block_shift] = out_buffer[
                                                                         :block_shift
                                                                         ]
    # cut audio to otiginal length
    predicted_speech = out_file[:len_audio]
    # check for clipping
    if np.max(predicted_speech) > 1:
        predicted_speech = predicted_speech / np.max(predicted_speech) * 0.99
    predicted_speech_reshape = predicted_speech.reshape(-1, 1)
    return predicted_speech_reshape, out_buffer, [states_1, states_2]


def inference_kws(kws_model, input_audio, audio_sample_length):
    audio_sample_length = int(audio_sample_length)
    if audio_sample_length > len(input_audio):
        padded_input_audio = np.pad(input_audio, ((0, int(audio_sample_length - len(input_audio))), (0, 0)), 'constant')
    else:
        padded_input_audio = input_audio[:int(audio_sample_length)]
    prediction = kws_model.predict(padded_input_audio.T.astype(np.float32))
    return prediction


def make_kws_results(kws_results, answer_labels, labels, score_option):
    score = 0.0
    if not kws_results:
        result = False
        print("sound is too small or non-voice sound(include voice at least 20%).")
        return score, result
    for answer_label in answer_labels:
        answer_results = []
        for kws_result in kws_results:
            kws_result = kws_result.squeeze()
            answer_result = kws_result[labels.index(answer_label)]
            answer_results.append(answer_result)
        answer_results = np.array(answer_results)
        top_results = np.argsort(-answer_results)[:3]
        topscore = 0.0
        for top_result in top_results:
            topscore += answer_results[top_result]
        topscore /= len(top_results)
        if score_option == "modified":
            if topscore >= 1.0:
                topscore = 1.0
            elif topscore >= 0.9:
                topscore = topscore ** 5
            elif topscore >= 0.8:
                topscore = topscore ** 4
            else:
                topscore = topscore ** 3
        score += (topscore)
    score = score / (len(answer_labels) * 1.0)
    if score > 1.0:
        score = 1.0
    elif score < 0.001:
        score = 0
    score = score * 100
    result = True
    return score, result


def find_answer_labels(keyword_sentence, labels):
    answer_labels = []
    for label in labels:
        if label in keyword_sentence:
            answer_labels.append(label)
    if "영화관" in answer_labels:
        answer_labels.remove("영화")
    return answer_labels


def run_model(audio, shift_duration=0.1, clip_duration=1.5, sample_rate=16000, vad_model=None, aec_model=None,
              kws_model=None):
    kws_results = []
    num_shift_frame = sample_rate * shift_duration
    num_clip_frame = clip_duration * sample_rate
    if audio.shape[0] < 0.3 * clip_duration * sample_rate:
        print("too short wav (~0.45sec)")
        return kws_results
    elif audio.shape[0] < 1.0 * clip_duration * sample_rate:
        num_frame = 1
    else:
        num_frame = math.ceil(
            (audio.shape[0] - (clip_duration - shift_duration) * sample_rate) / (num_shift_frame * 1.0))

    for i in range(num_frame):
        audio_frame = audio[int(i * num_shift_frame):int(i * num_shift_frame + num_clip_frame)]
        if vad_model is not None:
            vad_result = vad_function(sample_rate, 30, 0.2, vad_model, audio_frame)  # true or false
        else:
            vad_result = True
        if vad_result:
            kws_results.append(inference_kws(kws_model, audio_frame, num_clip_frame))

    return kws_results