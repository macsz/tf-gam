import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import scipy.misc
import numpy as np
from timeit import default_timer as timer


THRESHOLD_DOWN = 0.5
THRESHOLD_UP = 0.8


class ClassifyFace:
    _mat = {}
    _files = []
    _model_path = ''
    _label_lines = None
    noses = {}

    def get_color_intensity(self, prob, norm=False):
        max_val = 0.3
        min_val = 0.1
        if not norm:
            val = max_val * prob
        else:
            val = ((max_val-min_val)*(prob-THRESHOLD_DOWN))/(THRESHOLD_UP-THRESHOLD_DOWN)+min_val
        return val

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

    def run(self):
        self._load_tf()
        counter = 0
        failed_counter = 0
        with tf.Session() as sess:

            for image_path in self._files:
                counter += 1

                # Read in the image_data
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()

                time_start = timer()
                self._run_tf(sess, image_data)
                time_elapsed = timer() - time_start
                print('{0}/{1}'.format(counter, len(self._files)), 'Face TF elapsed time', image_path, time_elapsed)

                img_full = np.array(Image.open(image_path), dtype=np.uint8)

                # Create figure and axes
                fig, ax = plt.subplots(1)

                # Display the image
                ax.imshow(img_full)

                max_h = -1
                max_r = -1
                min_h = 999
                min_r = 999
                for h in range(0, 8):
                    for r in range(0, 8):
                        prob = self._mat[str(r) + ',' + str(h)]
                        if prob > THRESHOLD_DOWN:
                            tile = patches.Rectangle((80 / 8 * h, 60 / 8 * r), 80 / 8, 60 / 8, linewidth=1, edgecolor='g',
                                                     facecolor='g', alpha=self.get_color_intensity(prob, norm=False))
                            if h < min_h:
                                min_h = h
                            if r < min_r:
                                min_r = r
                            if h > max_h:
                                max_h = h
                            if r > max_r:
                                max_r = r
                        else:
                            tile = patches.Rectangle((80/8 * h, 60/8 * r), 80/8, 60/8, linewidth=1, edgecolor='r', facecolor='none',
                                                     alpha=0.5)
                        ax.add_patch(tile)

                detected_face = patches.Rectangle((80 / 8 * min_h, 60 / 8 * min_r), 80 / 8 * (max_h-min_h+1),
                                                  60 / 8 * (max_r-min_r+1), linewidth=3, edgecolor='g', facecolor='none')
                ax.add_patch(detected_face)
                img_face = img_full[min_r*int(60 / 8):(max_r+1)*int(60 / 8), min_h*int(80 / 8):(max_h+1)*int(80 / 8)]
                # plt.show()
                save_path = image_path.split('/')
                save_path[-2] = 'output_face'
                save_path = '/'.join(save_path)
                plt.savefig(save_path)

                nose_path = image_path.split('/')
                nose_path[-2] = 'input_nose'
                nose_path = '/'.join(nose_path)
                try:
                    scipy.misc.imsave(nose_path, img_face)
                except ValueError as ve:
                    print('FAILED', image_path)
                    failed_counter += 1

                nose_data = {
                    'orig_path': image_path,
                    'save_path': save_path,
                    'nose_path': nose_path,
                    'nose_position': {
                        'min_h': min_h,
                        'min_r': min_r,
                        'max_h': max_h,
                        'max_r': max_r
                    },
                    'data': img_face
                }
                self.noses[image_path] = nose_data
                plt.clf()
                plt.cla()
                plt.close('all')
        print('Total failed:', failed_counter)

    def __init__(self, files, model_path):
        self._model_path = model_path
        self._files = files
