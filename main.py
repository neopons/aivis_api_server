from aivis.wsgi import application
import model_web

app = application

vad_model = model_web.load_vad_model(3)
kws_model = model_web.load_kws_model('models_data/KWS/KCSC_child_v1.00/kwt3_softmax')
labels = model_web.read_labels('config/labels_KCSC_child_100.txt')