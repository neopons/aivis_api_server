import json
import time

from rest_framework.response import Response
from rest_framework.decorators import api_view
import model_web
import numpy as np
import base64
import scipy.io.wavfile as wav
import tempfile
import os
import main
# import psutil

# Create your views here.


# def memory_usage(message: str = 'debug'):
#     # current process RAM usage
#     p = psutil.Process()
#     rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
#     print(f"[{message}] memory usage: {rss: 10.5f} MB")


def base64_to_audio_numpy(audio_base64):
    audio_decoded = base64.b64decode(audio_base64)

    with tempfile.TemporaryDirectory() as tempDir:
        if os.path.exists(tempDir):

            wav_file = open(tempDir + "/audio.wav", "wb")
            wav_file.write(audio_decoded)
            samplerate, wave_data = wav.read(tempDir + "/audio.wav")

            audio = np.array(wave_data, np.float32).reshape(-1, 1) / 32768.0

            return audio

    return None


@api_view(['POST'])
def nonverbal(request):
    json_data = json.loads(request.body)

    audio_base64 = json_data['audio']

    audio = base64_to_audio_numpy(audio_base64)

    if audio is not None:

        start_time = time.time()

        # below nonverbal function results
        vad_model = model_web.load_vad_model(3)
        sample_rate = 16000

        vad_percent, voice_energy_list = model_web.run_model_nonverbal(audio, vad_model, sample_rate=sample_rate)
        print(f"success vad percent is {vad_percent}")
        print(f"success voice energies is {voice_energy_list}")

        end_time = time.time()

        print(f"Total Time : {end_time - start_time} sec")

        response = {
            'vad_percent': vad_percent,
            'voice_energy_list': voice_energy_list
        }

        return Response(response)

    else:
        response = {
            'vad_percent': 0,
            'voice_energy_list': []
        }

        return Response(response)


# @api_view(['POST'])
# def pronunciation(request):
#     start_time = time.time()
#
#     json_data = json.loads(request.body)
#
#     keyword = json_data['script']
#     audio_base64 = json_data['audio']
#
#     audio = base64_to_audio_numpy(audio_base64)
#
#     # 임시 코드
#     # audio = None
#     #
#
#     if audio is not None:
#
#         # if you want to reduce inference time, it is better to preload the kws_model.
#         vad_model = model_web.load_vad_model(3)
#
#         # if kws_model result always makes too high score, then change kws_model 'kwt3_softmax' to
#         # 'kwt3_softmax_nospecaug'
#         kws_model = model_web.load_kws_model('models_data/KWS/KCSC_child_v1.00/kwt3_softmax')
#         labels = model_web.read_labels('config/labels_KCSC_child_100.txt')
#
#         # if shift_duration value increases, total inference time decreases. but make more error.
#         shift_duration = 0.1  # seconds to shift
#         sample_rate = 16000
#         answer_labels = model_web.find_answer_labels(keyword, labels)
#
#         # if vad_model is None, then run_model function will run without vad_model
#         kws_results = model_web.run_model(audio, shift_duration=shift_duration, clip_duration=1.5, sample_rate=sample_rate,
#                                           vad_model=vad_model, kws_model=kws_model)
#
#         # if "raw" option score value is always 99~100, change option "modified"
#         score, result = model_web.make_kws_results(kws_results, answer_labels, labels, "raw")
#         print(f"final score : {score}")
#
#         end_time = time.time()
#
#         # print(f"Total Time : {end_time - start_time} sec")
#
#         # response = {'score': score}
#         response = {
#             'total_time': end_time - start_time,
#             'score': score
#         }
#
#         # memory_usage('pronunciation')
#
#         return Response(response)
#
#     else:
#         response = {'score': 0}
#
#         return Response(response)


@api_view(['POST'])
def pronunciation(request):
    start_time = time.time()

    json_data = json.loads(request.body)

    keyword = json_data['script']
    audio_base64 = json_data['audio']

    audio = base64_to_audio_numpy(audio_base64)

    shift_duration = 0.1  # seconds to shift
    sample_rate = 16000
    answer_labels = model_web.find_answer_labels(keyword, main.labels)

    kws_results = model_web.run_model(audio, shift_duration=shift_duration, clip_duration=1.5, sample_rate=sample_rate,
                                      vad_model=main.vad_model, kws_model=main.kws_model)

    score, result = model_web.make_kws_results(kws_results, answer_labels, main.labels, "raw")

    end_time = time.time()

    response = {
        'total_time': end_time - start_time,
        'score': score
    }

    return Response(response)


@api_view(['GET'])
def warmup(request):

    return Response()
