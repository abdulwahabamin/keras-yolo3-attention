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

f = open('log.txt','w+')

for i, layer in enumerate(model.layers):
    l = "Layer"+ str(i) + "\t"+ str(layer.name)+ "\t\t"+ str(layer.input_shape)+ "\t"+ str(layer.output_shape)
    print(l)
    f.write(l +'\n')
f.close()
print(len(model.layers))


from keras.utils import plot_model
plot_model(model, to_file='model.png')
print(model.inputs)