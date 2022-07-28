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

## Install
1. go to install_requirements directory
2. install install_requirements.sh (ex. $ sudo sh install requirements.sh)
3. if you use venv environment, setup venv environment and activate it
4. $ pip install -r requirements.txt

## Output details
* nonverbal case
  * output result : score, vad-percent 
  * score 값은 음성의 크기를 나타냄 즉 값이 클수록 크게 잘 발성 했음을 알 수 있음. 단, 주변 소음이 클수록 값도 크게 나오는 경향이 있음
  * score 값의 범위는 0에서 최대 수천까지 고려할 것. 실제 테스트시 0에서 수백까지 나오는 것 확인. 단, 테스트 기기와 환경에 따라 다를 수 있음.
  * score 값은 따로 크기를 0-100 으로 조정하지 않음. 스케일링 전에 실제 테스트를 통해서 값의 분포를 파악할 필요가 있고 최진호 대표님이 일단 의미 있는 값을 전달하는 것을 1차 목표로 하라 함.
  * vad-percent 값은 speech-length 시간 내에서 발성을 얼마나 했는지 퍼센트(%)로 결과 전달
  * ex. speech-length = 3 일때, vad-percent 값이 50이면 클라이언트가 보내준 3초의 소리 안에 음성이 50%인 1.5초가 포함되어 있다는 의미임
  * vad-percent 값의 범위는 0-100
* pronunciation case
  * output result : score
  * 이 경우 점수로 평가하여 전달
  * 단, 발화 문장에 아래 단어 중 하나가 들어가야함
  * 20220629 이후 사용 필수 포함 단어 목록 : 우리,동생,선생님,할아버지,사람,소리,노래,생일,머리,선물,냄새,무슨,삼촌,사진,다른,숙제,이름,바람,수업,다리,계속,연습,나중,아이스크림,그림,리코더,놀이,버스,수영,크리스마스
  * 현재 사용 모델 : 총 30개 단어 인식, 인식률 93.7% 이상 8만개이상의 유아 자유대화 데이터로 학습(타겟 단어 30가지, 비타겟 단어 30가지)
  * 그리고 발화 문장을 전달 할때는 구두점 등 글자 외 다른 문장 기호는 빼고 전달할 것
  * 원래 다른 용도로 쓰는 기술로 구현하여 실제 테스트 부족으로 score가 믿을 만하게 나오는지는 테스트 필요
  * 현재로써는 점수가 너무 높게 나오는 경향이 있어 결과값에 3-5제곱을 해서 제공 (실제 테스트 결과에 따라 조정하여 값 분포 조정)
  * score 값은 0-100 (너무 소리가 작거나 음성이 불분명할 경우 0점으로 나올 수 있음)
  

  * ~~20220629 이후 사용안함 (이전 필수 포함 단어 목록 : 영화관, 도서관, 주차, 영화, 예약, 어디, 확인, 시간, 대출, 자리)~~

## Run
we provide web server, testing client code with streaming audio
### Server Run
 $ python3 run_webserver.py

### Client Test
 $ python3 run_webclient.py