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

model.save_weights('model_log.h5')