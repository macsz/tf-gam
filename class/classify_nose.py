import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import operator
from timeit import default_timer as timer
from tools import get_color_intensity, get_cached_prob, get_avg_color, \
    array_slice

THRESHOLD_DOWN = 0.72
THRESHOLD_UP = 0.75


class ClassifyNose:
    _mat = {}
    _noses = []
    _model_path = ''
    _label_lines = None
    _sorted_probs = None

    def get_weights(self, shape, ):
        return tf.get_variable('weights', shape,
                               initializer=tf.zeros_initializer)

    def get_biases(self, shape):
        return tf.get_variable('biases', shape,
                               initializer=tf.zeros_initializer)

    def _load_tf(self):
        # Loads label file, strips off carriage return
        self._label_lines = [
            line.rstrip() for line
            in tf.gfile.GFile("class/models/nose_retrained_labels.txt")
        ]

        # Unpersists graph from file
        with tf.gfile.FastGFile(
                        "class/models/nose_retrained_graph.pb",
                        'rb'
        ) as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def _run_tf(self, sess, image_data):
        # Feed the image_data as input to the graph and get first prediction

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        feature_tensor = sess.graph.get_tensor_by_name('mixed_10/join:0')
        # feature_set: 1x 8x8x2048
        feature_set = sess.run(feature_tensor,
                               {'DecodeJpeg/contents:0': image_data})

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
        self._sorted_probs = sorted(self._mat.items(),
                                    key=operator.itemgetter(1))

    def run(self):
        self._load_tf()
        counter = 0
        cache = []
        all_frames_cells_avg_color_sum = 0
        avg_successful_frames = len(self._noses.keys())
        with tf.Session() as sess:
            # For cache to work dict must be sorted
            for image_path in sorted(self._noses.keys()):
                counter += 1
                try:
                    image_data = tf.gfile.FastGFile(
                        self._noses[image_path]['nose_path'], 'rb').read()
                except:
                    continue

                time_start = timer()
                self._run_tf(sess, image_data)
                time_elapsed = timer() - time_start
                print('{0}/{1}'.format(counter, len(self._noses)),
                      'Nose TF elapsed time', image_path, time_elapsed)

                img_full_for_nose = np.array(
                    Image.open(self._noses[image_path]['nose_path']),
                    dtype=np.uint8)
                img_full = np.array(
                    Image.open(self._noses[image_path]['orig_path']),
                    dtype=np.uint8)
                lx, ly = len(img_full_for_nose[0]), len(img_full_for_nose)
                xp = lx/8
                yp = ly/8

                # Create figure and axes
                fig, ax = plt.subplots(1)

                # Display the image
                ax.imshow(img_full)

                frame_cache = [
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
                    for y in range(0, 8):
                        prob = self._mat[str(y) + ',' + str(h)]
                        if prob > THRESHOLD_DOWN and prob in [x[1] for x in self._sorted_probs[-2:]]:
                            frame_cache[h][y] = 1

                cache.append(frame_cache)

                max_x = -1
                max_y = -1
                min_x = 999
                min_y = 999

                cells_avg_color = []

                for x in range(0, 8):
                    for y in range(0, 8):
                        prob = self._mat[str(y) + ',' + str(x)]
                        # if prob > THRESHOLD_DOWN:
                        cached_prob, activity = get_cached_prob(
                            cache=cache, x=x, y=y, active_frames=10,
                            threshold=4)
                        if cached_prob and activity > 0:
                            if xp * x + self._noses[image_path]['nose_position']['min_x'] * 80 / 8 < min_x:
                                min_x = xp * x + self._noses[image_path]['nose_position']['min_x'] * 80 / 8
                            if yp * y + self._noses[image_path]['nose_position']['min_y'] * 60 / 8 < min_y:
                                min_y = yp * y + self._noses[image_path]['nose_position']['min_y'] * 60 / 8
                            if xp * x + self._noses[image_path]['nose_position']['min_x'] * 80 / 8 + xp \
                                    > max_x:
                                max_x = xp * x + self._noses[image_path]['nose_position']['min_x'] * 80 / 8 + xp
                            if yp * y + self._noses[image_path]['nose_position']['min_y'] * 60 / 8 + yp > max_y:
                                max_y = yp * y + self._noses[image_path]['nose_position']['min_y'] * 60 / 8 + yp
                            tile = patches.Rectangle(
                                (xp * x + self._noses[image_path]['nose_position']['min_x'] * 80 / 8,
                                 yp * y + self._noses[image_path]['nose_position']['min_y'] * 60 / 8),
                                xp, yp, linewidth=1,
                                edgecolor='b', facecolor='b',
                                alpha=get_color_intensity(prob, norm=False,
                                    threshold_up=THRESHOLD_UP,
                                    threshold_down=THRESHOLD_DOWN)
                            )
                            ax.add_patch(tile)

                            avg_active_color = get_avg_color(array_slice(
                                img_full, x=int(xp * x), y=int(yp * y),
                                w=int(xp), h=int(yp)))
                            cells_avg_color.append(avg_active_color)
                            # text_h = lh / 8 * h + self._noses[image_path]['nose_position']['min_h'] * 80 / 8 + lh/16
                            # text_r = yp * y + self._noses[image_path]['nose_position']['min_y'] * 60 / 8 + ly/16
                            # ax.annotate(str(activity),
                            #     xytext=(
                            #         text_h,
                            #         text_r
                            #     ),
                            #     xy=(
                            #         text_h,
                            #         text_r
                            #     ))
                frame_active_avg = np.average(cells_avg_color)
                import math
                if not math.isnan(frame_active_avg):
                    all_frames_cells_avg_color_sum += frame_active_avg
                else:
                    frame_active_avg = None
                    avg_successful_frames -= 1
                print('Nose (frame) average color for active cells:',
                      frame_active_avg)

                tile = patches.Rectangle(
                    (self._noses[image_path]['nose_position']['min_x'] * 80 / 8,
                     self._noses[image_path]['nose_position']['min_y'] * 60 / 8),
                    lx, ly, linewidth=3, edgecolor='g', facecolor='none',
                    alpha=0.5)
                ax.add_patch(tile)

                detected_nose = patches.Rectangle(
                    (min_x, min_y), (max_x-min_x), (max_y-min_y),
                    linewidth=2, edgecolor='b', facecolor='none', alpha=0.5)
                ax.add_patch(detected_nose)

                save_path = image_path.split('/')
                save_path[-2] = 'output_nose'
                save_path = '/'.join(save_path)
                plt.savefig(save_path)
                plt.clf()
                plt.cla()
                plt.close('all')
        print('Nose (movie) average color for active cells:',
              (all_frames_cells_avg_color_sum / avg_successful_frames))

    def __init__(self, noses):
        self._noses = noses
