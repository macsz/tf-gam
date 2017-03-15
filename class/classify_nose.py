import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import operator
from timeit import default_timer as timer
from tools import get_color_intensity, get_cached_prob

THRESHOLD_DOWN = 0.72
THRESHOLD_UP = 0.75


class ClassifyNose:
    _mat = {}
    _noses = []
    _model_path = ''
    _label_lines = None
    _sorted_probs = None

    def get_weights(self, shape, ):
        tf.get_variable('weights', shape, initializer=tf.zeros_initializer)
        return tf.get_variable('weights', shape, initializer=tf.zeros_initializer)

    def get_biases(self, shape):
        return tf.get_variable('biases', shape, initializer=tf.zeros_initializer)

    def _load_tf(self):
        # Loads label file, strips off carriage return
        self._label_lines = [line.rstrip() for line in tf.gfile.GFile(self._model_path + "retrained_labels.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile(self._model_path + "retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def _run_tf(self, sess, image_data):
        # Feed the image_data as input to the graph and get first prediction

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        feature_tensor = sess.graph.get_tensor_by_name('mixed_10/join:0')
        # feature_set: 1x 8x8x2048
        feature_set = sess.run(feature_tensor, {'DecodeJpeg/contents:0': image_data})

        for cells in feature_set:
            for x in range(0, len(cells)):
                for y in range(0, len(cells[x])):
                    cell = cells[x][y]
                    cell = np.reshape(cell, (1, 1, 1, 2048))
                    a = sess.run(softmax_tensor, {'pool_3:0': cell})

                    probabilities = a[0]
                    top_k = probabilities.argsort()[-len(probabilities):][::-1]
                    for node_id in top_k:
                        human_string = self._label_lines[node_id]
                        score = probabilities[node_id]
                        if human_string != 'other':
                            self._mat[str(x) + ',' + str(y)] = score
        self._sorted_probs = sorted(self._mat.items(), key=operator.itemgetter(1))

    def run(self):
        self._load_tf()
        counter = 0
        cache = []
        with tf.Session() as sess:
            # For cache to work dict must be sorted
            for image_path in sorted(self._noses.keys()):
                counter += 1
                try:
                    image_data = tf.gfile.FastGFile(self._noses[image_path]['nose_path'], 'rb').read()
                except:
                    continue

                time_start = timer()
                self._run_tf(sess, image_data)
                time_elapsed = timer() - time_start
                print('{0}/{1}'.format(counter, len(self._noses)), 'Nose TF elapsed time', image_path, time_elapsed)

                img_full_for_nose = np.array(Image.open(self._noses[image_path]['nose_path']), dtype=np.uint8)
                img_full = np.array(Image.open(self._noses[image_path]['orig_path']), dtype=np.uint8)
                lh, lr = len(img_full_for_nose[0]), len(img_full_for_nose)

                # Create figure and axes
                fig, ax = plt.subplots(1)

                # Display the image
                ax.imshow(img_full)

                cache_frame = [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]

                for h in range(0, 8):
                    for r in range(0, 8):
                        prob = self._mat[str(r) + ',' + str(h)]
                        if prob > THRESHOLD_DOWN:

                            if prob in [x[1] for x in self._sorted_probs[-2:]]:
                                tile = patches.Rectangle(
                                    (lh / 8 * h + self._noses[image_path]['nose_position']['min_h'] * 80 / 8,
                                     lr / 8 * r + self._noses[image_path]['nose_position']['min_r'] * 60 / 8),
                                    lh / 8, lr / 8, linewidth=1,
                                    edgecolor='b',
                                    facecolor='b',
                                    alpha=get_color_intensity(prob, norm=False,
                                                              threshold_up=THRESHOLD_UP,
                                                              threshold_down=THRESHOLD_DOWN))
                                ax.add_patch(tile)
                                cache_frame[h][r] += 1

                            if get_cached_prob(cache=cache, h=h, r=r):
                                tile = patches.Rectangle(
                                    (lh / 8 * h + self._noses[image_path]['nose_position']['min_h'] * 80 / 8,
                                     lr / 8 * r + self._noses[image_path]['nose_position']['min_r'] * 60 / 8),
                                    lh / 8, lr / 8, linewidth=1,
                                    edgecolor='r',
                                    facecolor='r',
                                    alpha=get_color_intensity(prob, norm=False,
                                                              threshold_up=THRESHOLD_UP,
                                                              threshold_down=THRESHOLD_DOWN))
                                ax.add_patch(tile)
                cache.append(cache_frame)

                tile = patches.Rectangle(
                    (self._noses[image_path]['nose_position']['min_h'] * 80 / 8,
                     self._noses[image_path]['nose_position']['min_r'] * 60 / 8),
                    lh, lr, linewidth=1, edgecolor='g',
                    facecolor='none',
                    alpha=0.5)
                ax.add_patch(tile)

                save_path = image_path.split('/')
                save_path[-2] = 'output_nose'
                save_path = '/'.join(save_path)
                plt.savefig(save_path)
                plt.clf()
                plt.cla()
                plt.close('all')

    def __init__(self, noses, model_path):
        self._model_path = model_path
        self._noses = noses
