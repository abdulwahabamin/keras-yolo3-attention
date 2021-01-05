import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import tensorflow as tf

from yolo3.model import yolo_body as yolo_body


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


model_path = 'logs/yolo-attention-log-multiply/trained_weights_final.h5'
image_path = '/Ted/datasets/Garbage/VOC_Test_Hard/JPEGImages/im187.jpg'
anchors_path = 'model_data/yolo_anchors.txt'

K.set_learning_phase(0)

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
yolo_model = Model(yolo_model.input, yolo_model.output)
yolo_model.compile(loss="mse", optimizer="adam")
print(yolo_model.summary())

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in yolo_model.outputs])
tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)
# # size = 416, 416
# X = Image.open(image_path)
# X = X.resize((416, 416), Image.BICUBIC)
# X = np.array(X, dtype='float32')
# # print(X)
