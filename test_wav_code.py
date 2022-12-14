# import base64

import model_web
import numpy as np
import scipy.io.wavfile as wav
# import sys

import time
import psutil


def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")


if __name__ == '__main__':

    memory_usage('__main__')

    start_time = time.time()

    # change path and keyword arguments
    wav_file = "sample_wavs/peter.wav"
    # keyword = "바람과 함께 사라지다"
    keyword = "선생님"

    result = False
    score = 0.0

    # base64Encoded = base64.b64encode(open(wav_file, "rb").read())
    # print(f"base64Encoded : {base64Encoded}")
    # audio_decoded = base64.b64decode(base64Encoded)
    # print(f"audio_decoded : {audio_decoded.hex()}")
    
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
    # if kws_model result always makes too high score, then change kws_model 'kwt3_softmax' to 'kwt3_softmax_nospecaug'
    kws_model = model_web.load_kws_model('models_data/KWS/KCSC_child_v1.00/kwt3_softmax')
    labels = model_web.read_labels('config/labels_KCSC_child_100.txt')

    memory_usage('load vad kws labels')

    #if shift_duration value increases, total inference time decreases. but make more error.
    shift_duration = 0.1  # seconds to shift
    sample_rate = 16000

    for i in range(0, 100):
        answer_labels = model_web.find_answer_labels(keyword, labels)

        #if vad_model is None, then run_model function will run without vad_model
        # if sound is too small or non-voice sound(include voice at least 20%), then output is False.
        kws_results = model_web.run_model(audio, shift_duration=shift_duration, clip_duration=1.5, sample_rate=sample_rate, vad_model=vad_model, kws_model=kws_model)

        # if you want to make result in small sound, then change load_vad_model(1 or 2). but in that case, model becomes more sensitive to noise.
        # if "raw" option score value is always 99~100, change option "modified"
        score, result = model_web.make_kws_results(kws_results, answer_labels, labels, "raw")
        print(f"final score : {score}")

        memory_usage('find lanels, run model, make kws results')

    ############## below nonverbal function results ###############
    # vad_model = model_web.load_vad_model(3)
    # sample_rate = 16000
    #
    # vad_percent, voice_energy_list = model_web.run_model_nonverbal(audio, vad_model, sample_rate=sample_rate)
    # print(f"success vad percent is {vad_percent}")
    # print(f"success voice energies is {voice_energy_list}")

    end_time = time.time()

    print(f"Total Time : {end_time - start_time} sec")
