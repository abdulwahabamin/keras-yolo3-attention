# from quiver_engine import server
# from keras.models import load_model

# model = load_model('model_log.h5')

# server.launch(
#     model, # a Keras Model

#     ['garbage'], # list of output classes from the model to present (if not specified 1000 ImageNet classes will be used)

#     # where to store temporary files generatedby quiver (e.g. image files of layers)
#     '/Ted/models/keras-yolo3-attention/activations',

#     # a folder where input images are stored
#     '/Ted/datasets/Garbage/VOC_Test_Easy',

#     # the localhost port the dashboard is to be served on
#     5000
# )
from yolo3.model import yolo_body
import keras
from keras.layers import Input
import numpy as np


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

anchors_path = 'model_data/yolo_anchors.txt'
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)
image_input = Input(shape=(None, None, 3))
model = yolo_body(image_input,num_anchors//3,1)

model.load_weights('logs/yolo-attention-log/trained_weights_final.h5')


from keract import get_activations
from PIL import Image
img  = '/Ted/datasets/Garbage/VOC_Test_Easy/JPEGImages/im183.jpg'
image = Image.open(img)
activations = get_activations(model, image)

from keract import display_heatmaps
display_activations(activations, save=True)