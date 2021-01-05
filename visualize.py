from keras.layers import Input
from keras.models import Model
import numpy as np
from yolo3.model import yolo_body as yolo_body
#from yolo3.model import yolo_body
import keract
from PIL import Image
model_path = 'logs/yolo-attention-log-multiply/trained_weights_final.h5'
image_path = '/Ted/datasets/Garbage/VOC_Test_Hard/JPEGImages/im187.jpg'
anchors_path = 'model_data/yolo_anchors.txt'


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

anchors = get_anchors(anchors_path)
num_anchors = len(anchors)

yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes=1)
yolo_model.load_weights(model_path)
yolo_model = Model(yolo_model.input, yolo_model.layers[50].output)
yolo_model.compile(loss="mse", optimizer="adam")
print(yolo_model.summary())

# size = 416, 416
X = Image.open(image_path)
X = X.resize((416, 416), Image.BICUBIC)
X = np.array(X, dtype='float32')
# print(X)
X /= 255.
X = np.expand_dims(X, 0)  # Add batch dimension.
activations_rpn = keract.get_activations(yolo_model, X)
dp_activations = keract.display_activations(activations_rpn, cmap='gray',
                                            directory='activations/attention', save=True)