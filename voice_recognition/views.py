import json
from rest_framework.response import Response
from rest_framework.decorators import api_view
import model_web
import numpy as np
import base64
import scipy.io.wavfile as wav


# Create your views here.
@api_view(['POST'])
def recognition(request):
    json_data = json.loads(request.body)

    keyword = json_data['script']
    audio_base64 = json_data['audio']
    # print(f"audio_base64 : {audio_base64}")
    audio_decoded = base64.b64decode(audio_base64)
    # print(f"audio_decoded : {audio_decoded.hex()}")
    # print(f"audio_base64 byte array : {bytearray(audio_decoded)}")
    wav_file = open("audio.wav", "wb")
    wav_file.write(audio_decoded)
    samplerate, wave_data = wav.read("audio.wav")
    # print(f"wave_data : {wave_data}")

    # audio_decoded_list = list(audio_decoded)
    # print(f"audio_decoded_list : {audio_decoded_list}")

    audio = np.array(wave_data, np.float32).reshape(-1, 1) / 32768.0
    # print(f"audio : {audio}")

    # if you want to reduce inference time, it is better to preload the kws_model.
    vad_model = model_web.load_vad_model(3)
    kws_model = model_web.load_kws_model('models_data/KWS/KCSC_child_v0.01/kwt3_softmax')
    labels = model_web.read_labels('config/labels_KCSC_child_30.txt')

    # if shift_duration value increases, total inference time decreases. but make more error.
    shift_duration = 0.1  # seconds to shift
    sample_rate = 16000
    answer_labels = model_web.find_answer_labels(keyword, labels)

    # if vad_model is None, then run_model function will run without vad_model
    kws_results = model_web.run_model(audio, shift_duration=shift_duration, clip_duration=1.5, sample_rate=sample_rate,
                                      vad_model=vad_model, kws_model=kws_model)

    # if "raw" option score value is always 99~100, change option "modified"
    score, result = model_web.make_kws_results(kws_results, answer_labels, labels, "raw")
    print(f"final score : {score}")

    response = {'score': score}

    return Response(response)
