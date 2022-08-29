# Web server for VAD, KWS, AEC

## Updates
* 20220621
  * first update for VAD (especially nonverbal function)
  * update README.md
* 20220623
  * upload score system for pronunciation function
  * change server mechanism : 1회성 동작 말고 지속 동작 가능
* 20220624
  * add score result for nonverbal function
* 20220629
  * update new model for pronunciation function (check Output details)
  * change code model.py, make_kws_results function in run_webserver
* 20220724
  * update webfunction code, move original code in webserver directory.
* 20220725
  * update dataset, sample_code, sample_wavs, test_wav_code.
* 20220803
  * update nonverbal function code in model_web.py in webfunction directory.
  * change README explanation for nonverbal function
* 20220827
  * update 100 words list, dataset, and two models.
  * Please check sample code in webfunction dir.
  * Please check below Output details for difference of two models and 100 word list.

## Install
1. go to install_requirements directory
2. install install_requirements.sh (ex. $ sudo sh install requirements.sh)
3. if you use venv environment, setup venv environment and activate it
4. $ pip install -r requirements.txt

## Output details
* nonverbal case
  * output result : vad-percent, voice energy list
  * vad-percent 값은 speech-length 시간 내에서 발성을 얼마나 했는지 퍼센트(%)로 결과 전달
  * ex. speech-length = 3 일때, vad-percent 값이 50이면 클라이언트가 보내준 3초의 소리 안에 음성이 50%인 1.5초가 포함되어 있다는 의미임
  * vad-percent 값의 범위는 0-100 (%)
  * voice energy list is Python List of voice energies
  * sum of energy in frequency band (300 Hz to 7200 Hz)
  * window_size_ms : 40ms, no overlapping
  * range of energy value : 0 - infinite ( value 0 in deactivated VAD section)
* pronunciation case
  * output result : score
  * 이 경우 점수로 평가하여 전달
  * 단, 발화 문장에 아래 단어 중 하나가 들어가야함
  * 20220827 이후 사용 필수 포함 100 단어 목록 : 우리,동생,선생님,할아버지,사람,소리,노래,생일,머리,선물,냄새,무슨,삼촌,사진,다른,숙제,이름,바람,수업,다리,계속,연습,아이스크림,그림,리코더,놀이,버스,수영,크리스마스,여름,사과,풍선,놀이터,지렁이,도서관,수박,벌써,코끼리,생각,마리,수영장,인사,고래,청소,이사,수도,선수,색깔,빨간색,우산,주스,수학,책상,주사,산타,스티커,어린이,젖소,마스크,개구리,세상,사탕,약속,사랑,요리,소원,모래,하루,얼음,바이올린,미술,호랑이,자리,수건,차례,구름,휘파람,무릎,살살,상자,그릇,여섯,코로나,산책,다섯,새끼,원숭이,레고,의사,백설공주,노란색,손가락,우리나라,파란색,고슴도치,엘리베이터,초록색,쓰레기,어린이날,샌드위치
  * 그리고 발화 문장을 전달 할때는 구두점 등 글자 외 다른 문장 기호는 빼고 전달할 것
  * 원래 다른 용도로 쓰는 기술로 구현하여 실제 테스트 부족으로 score가 믿을 만하게 나오는지는 테스트 필요
  * 현재로써는 점수가 너무 높게 나오는 경향이 있어 결과값에 3-5제곱을 해서 제공 (실제 테스트 결과에 따라 조정하여 값 분포 조정)
  * score 값은 0-100 (너무 소리가 작거나 음성이 불분명할 경우 0점으로 나올 수 있음)
  * model 1 (KCSC_child_v1.00/kwt3_softmax)
    * 모델 : 총 100개 단어 인식, 인식률 94.1% 이상 18만개이상의 유아 자유대화 데이터로 학습(타겟 단어 100가지, 비타겟 단어 30가지)
    * description : use specaug
    * advantages : high recognition rate, high resistance to noise(__maybe__)
    * disadvantages : too high output score (overfitting problems), high false positive score(__maybe__)
  * model 2 (KCSC_child_v1.00/kwt3_softmax_nospecaug)
    * 모델 : 총 100개 단어 인식, 인식률 91.7% 이상 18만개이상의 유아 자유대화 데이터로 학습(타겟 단어 100가지, 비타겟 단어 30가지)
    * description : do not use specaug
    * advantages :  proper output score(__maybe__), low false positive score(__maybe__)
    * disadvantages : low recognition rate, low resistance to noise
  * __advantages and disadvantages marked (maybe) should be checked before using this model in real apps.__

  
  ~~* 20220629 이후 사용 필수 포함 30 단어 목록 : 우리,동생,선생님,할아버지,사람,소리,노래,생일,머리,선물,냄새,무슨,삼촌,사진,다른,숙제,이름,바람,수업,다리,계속,연습,나중,아이스크림,그림,리코더,놀이,버스,수영,크리스마스~~
  ~~* 20220629 이후 사용 모델 : 총 30개 단어 인식, 인식률 93.7% 이상 8만개이상의 유아 자유대화 데이터로 학습(타겟 단어 30가지, 비타겟 단어 30가지)~~
  ~~* 20220629 이후 사용안함 (이전 필수 포함 단어 목록 : 영화관, 도서관, 주차, 영화, 예약, 어디, 확인, 시간, 대출, 자리)~~

## Run
we provide web server, testing client code with streaming audio
### Server Run
 $ python3 run_webserver.py

### Client Test
 $ python3 run_webclient.py