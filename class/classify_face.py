import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import scipy.misc
import numpy as np
from timeit import default_timer as timer
import tools

THRESHOLD_DOWN = 0.5
THRESHOLD_UP = 0.8


class ClassifyFace:
    _mat = {}
    _files = []
    _model_path = ''
    _label_lines = None
    noses = {}

    # SAD
    sad_frame_sum = 0.0
    sad_previous_frame = np.array([])
    # SAD END

    def sad(self, frame):
        """
        public static double getMovementCoef(ArrayList<BufferedImage> movie){
            double frameSum = 0.0;
            for(int frameId=1; frameId<movie.size(); frameId++){
                BufferedImage frame1 = movie.get(frameId-1);
                BufferedImage frame2 = movie.get(frameId);
                frameSum += substractFrames(frame1, frame2);
            }
            return frameSum / movie.size();
        }
        
        private static double substractFrames(BufferedImage frame1, BufferedImage frame2){
            double pixelSum = 0.0;
            int pixel1, pixel2, pixel1Red, pixel2Red;
            for(int x=0; x<frame1.getWidth(); x++){
                for(int y=0; y<frame1.getHeight(); y++){
                    pixel1 = frame1.getRGB(x, y);
                    pixel2 = frame2.getRGB(x, y);
                    pixel1Red = (pixel1 >> 16) & 0x000000FF;
                    pixel2Red = (pixel2 >> 16) & 0x000000FF;
    
                    pixelSum += Math.abs(pixel2Red - pixel1Red);
                }
            }
            return pixelSum / frame1.getWidth() / frame1.getHeight();
        }
        :param frame:
        :return: 
        """
        def substract_frames(frame1, frame2):
            pixel_sum = 0
            x_range = len(frame1)
            y_range = len(frame1[0])
            for x in range(x_range):
                for y in range(y_range):
                    c1 = frame1[x][y][0]
                    c2 = frame2[x][y][0]
                    import math
                    pixel_sum += math.fabs(c1-c2)
            return pixel_sum / x_range / y_range

        if not self.sad_previous_frame.any():
            self.sad_previous_frame = frame
            return

        self.sad_frame_sum += substract_frames(self.sad_previous_frame, frame)

    def get_sad(self):
        return self.sad_frame_sum / len(self._files)

    def get_weights(self, shape):
        return tf.get_variable('weights', shape,
                               initializer=tf.zeros_initializer)

    def get_biases(self, shape):
        return tf.get_variable('biases', shape,
                               initializer=tf.zeros_initializer)

    def _load_tf(self):
        # Loads label file, strips off carriage return
        self._label_lines = [
            line.rstrip() for line
            in tf.gfile.GFile("class/models/face_retrained_labels.txt")
        ]

        # Unpersists graph from file
        with tf.gfile.FastGFile("class/models/face_retrained_graph.pb", 'rb')\
                as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def _run_tf(self, sess, image_data):
            # Feed the image_data as input to the graph and get first
            # prediction

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

    def run(self):
        self._load_tf()
        counter = 0
        cache = []
        failed_counter = 0
        all_frames_cells_avg_color_sum = 0
        xp = 80 / 8
        yp = 60 / 8
        face_coords_static = self._params['face_coords_static']
        static_movie_cells_avg = []
        with tf.Session() as sess:

            for image_path in sorted(self._files):
                counter += 1

                # Read in the image_data
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()

                time_start = timer()
                self._run_tf(sess, image_data)
                time_elapsed = timer() - time_start
                print('{0}/{1}'.format(counter, len(self._files)),
                      'Face TF elapsed time', image_path, time_elapsed)

                img_full = np.array(Image.open(image_path), dtype=np.uint8)

                # count SAD
                self.sad(img_full)

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
                frame_mask = [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]

                for x in range(0, 8):
                    for y in range(0, 8):
                        prob = self._mat[str(y) + ',' + str(x)]
                        if prob > THRESHOLD_DOWN:
                            frame_cache[x][y] += 1

                cache.append(frame_cache)

                for x in range(0, 8):
                    for y in range(0, 8):
                        prob = self._mat[str(y) + ',' + str(x)]
                        if prob > THRESHOLD_DOWN:
                            cached_prob, activity = tools.get_cached_prob(
                                cache=cache, x=x, y=y)
                            if cached_prob:
                                frame_mask[x][y] = 1
                max_x = -1
                max_y = -1
                min_x = 999
                min_y = 999
                cells_avg_color = []
                static_active_overlay_count = 0
                active_cells_count = 0
                static_x1 = face_coords_static['x1']
                static_x2 = face_coords_static['x2']
                static_y1 = face_coords_static['y1']
                static_y2 = face_coords_static['y2']
                static_frame_cells_avg = []

                for x in range(0, 8):
                    if tools.get_column_power(frame_mask, x) > 1:
                        for y in range(0, 8):
                            # ## avg color for static frame
                            if static_x1 <= x <= static_x2 and \
                                                    static_y1 <= y <= static_y2:
                                static_frame_cells_avg.append(
                                    tools.get_avg_color(
                                        tools.array_slice(img_full,
                                                        x=int(xp * x),
                                                        y=int(yp * y),
                                                        w=int(xp),
                                                        h=int(yp)
                                                        )))
                            # ## END avg color for static frame
                            prob = self._mat[str(y) + ',' + str(x)]
                            if prob > THRESHOLD_DOWN:
                                if x < min_x:
                                    min_x = x
                                if y < min_y:
                                    min_y = y
                                if x > max_x:
                                    max_x = x
                                if y > max_y:
                                    max_y = y
                                tile = patches.Rectangle(
                                    (xp*x, yp*y),
                                    xp, yp,
                                    linewidth=1, edgecolor='g', facecolor='g',
                                    alpha=tools.get_color_intensity(prob,
                                                                    norm=False)
                                )
                                ax.add_patch(tile)

                                cells_avg_color.append(tools.get_avg_color(
                                    tools.array_slice(img_full,
                                                      x=int(xp*x),
                                                      y=int(yp*y),
                                                      w=int(xp),
                                                      h=int(yp)
                                                      )))

                                active_cells_count += 1
                                if static_x1 <= x <= \
                                        static_x2 and \
                                        static_y1 <= y <= static_y2:
                                    static_active_overlay_count += 1
                            else:
                                # draw red rectangles for non-active areas #1/2
                                tile = patches.Rectangle(
                                    (xp * x, yp * y),
                                    xp, yp,
                                    linewidth=1, edgecolor='r',
                                    facecolor='none', alpha=0.5)
                                ax.add_patch(tile)
                    else:
                        # draw red rectangles for non-active areas #2/2
                        for y in range(0, 8):
                            tile = patches.Rectangle(
                                (xp * x, yp * y),
                                xp, yp,
                                linewidth=1, edgecolor='r', facecolor='none',
                                alpha=0.5)
                            ax.add_patch(tile)
                static_frame_cells_avg = np.average(static_frame_cells_avg)
                print('Face (frame) static cells avg color:',
                      static_frame_cells_avg)
                static_movie_cells_avg.append(static_frame_cells_avg)
                print('Face (frame) active cells count:',
                      active_cells_count)
                print('Face (frame) cells overlaying with static count:',
                      static_active_overlay_count)
                frame_active_avg = np.average(cells_avg_color)
                print('Face (frame) average color for active cells:',
                      frame_active_avg)
                all_frames_cells_avg_color_sum += frame_active_avg

                detected_face = patches.Rectangle(
                    (80/8*min_x, 60/8*min_y),
                    80/8*(max_x-min_x+1),
                    60/8*(max_y-min_y+1),
                    linewidth=3, edgecolor='g', facecolor='none')
                ax.add_patch(detected_face)
                img_face = img_full[
                           min_y*int(60/8):(max_y+1)*int(60/8),
                           min_x*int(80/8):(max_x+1)*int(80/8)]
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
                        'min_x': min_x,
                        'min_y': min_y,
                        'max_x': max_x,
                        'max_y': max_y
                    },
                    'data': img_face
                }
                self.noses[image_path] = nose_data
                plt.clf()
                plt.cla()
                plt.close('all')
        print('Face (movie) static cells avg color:',
              np.average(static_movie_cells_avg))
        print('Face (movie) average color for active cells:',
              (all_frames_cells_avg_color_sum/len(self._files)))
        print('Face (movie) SAD value:', self.get_sad())
        print('Total failed:', failed_counter)

    def __init__(self, files, params):
        self._files = files
        self._params = params
