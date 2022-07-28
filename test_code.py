import json
from random import sample

from rest_framework.response import Response
from rest_framework.decorators import api_view
import model_web
import numpy as np
import base64
# import json


# Create your views here.
@api_view(['POST'])
def recognition(request):

    json_data = json.loads(request.body)

    result = False
    score = 0.0

    keyword = json_data['script']
    audio_base64 = json_data['audio']
    audio_decoded_list = list(base64.b64decode(audio_base64))
    if len(audio_decoded_list) != 0:
        audio = np.array(audio_decoded_list, np.float32).reshape(-1, 1)

        #if you want to reduce inference time, it is better to preload the kws_model.
        vad_model = model_web.load_vad_model(3)
        kws_model = model_web.load_kws_model('models_data/KWS/KCSC_child_v0.01/kwt3_softmax')
        labels = model_web.read_labels('config/labels_KCSC_child_30.txt')

        #if shift_duration value increases, total inference time decreases. but make more error.
        shift_duration = 0.1  # seconds to shift
        sample_rate = 16000
        answer_labels = model_web.find_answer_labels(keyword, labels)
        
        #if vad_model is None, then run_model function will run without vad_model
        kws_results=model_web.run_model(audio, shift_duration=shift_duration, clip_duration=1.5, sample_rate=sample_rate, vad_model=vad_model, kws_model=kws_model)
        
        # if "raw" option score value is always 99~100, change option "modified"
        score = model_web.make_kws_results(kws_results, answer_labels, labels, "raw")
        print(f"final score : {score}")
        result = True

    response = {}
    if result:
        response['score'] = score

    return Response(response)
