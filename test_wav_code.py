import base64

import model_aivis as model_web
import numpy as np
import scipy.io.wavfile as wav
import sys

if __name__ == '__main__':
    # change path and keyword arguments
    wav_file = "sample_wavs/peter.wav"
    # keyword = "바람과 함께 사라지다"
    keyword = "선생님"

    result = False
    score = 0.0

    base64Encoded = base64.b64encode(open(wav_file, "rb").read())
    # print(f"base64Encoded : {base64Encoded}")
    audio_decoded = base64.b64decode(base64Encoded)
    print(f"audio_decoded : {audio_decoded.hex()}")
    
    #wavread code
    samplerate, wave_data = wav.read(wav_file)
    # np.set_printoptions(threshold=sys.maxsize)
    print(f"wave_data : {wave_data}")
    # Normalize short ints to floats in range [-1..1).
    audio = np.array(wave_data, np.float32).reshape(-1, 1) / 32768.0
    # print(f"audio : {audio}")
    if samplerate != 16000:
        print("samplerate is not 16000")
        exit()

    ########## below this line is exactly the same as sample_code.py ###########
    #if you want to reduce inference time, it is better to preload the kws_model.
    vad_model = model_web.load_vad_model(3)
    kws_model = model_web.load_kws_model('models_data/KWS/KCSC_child_v0.01/kwt3_softmax')
    labels = model_web.read_labels('config/labels_KCSC_child_30.txt')

    #if shift_duration value increases, total inference time decreases. but make more error.
    shift_duration = 0.1  # seconds to shift
    sample_rate = 16000
    answer_labels = model_web.find_answer_labels(keyword, labels)
    
    #if vad_model is None, then run_model function will run without vad_model
    # if sound is too small or non-voice sound(include voice at least 20%), then output is False.
    kws_results=model_web.run_model(audio, shift_duration=shift_duration, clip_duration=1.5, sample_rate=sample_rate, vad_model=None, kws_model=kws_model)
    
    if not kws_results:
        result = False
        score = 0.0
        print("sound is too small or non-voice sound(include voice at least 20%).")
        # if you want to make result in small sound, then change load_vad_model(1 or 2). but in that case, model becomes more sensitive to noise.
    else:
        # if "raw" option score value is always 99~100, change option "modified"
        score = model_web.make_kws_results(kws_results, answer_labels, labels, "raw")
        print(f"final score : {score}")
        result = True

