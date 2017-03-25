import argparse
import tools
from classify_face import ClassifyFace
from classify_nose import ClassifyNose
from os import listdir
from os.path import isfile, join

ROOT_PATH = '/home/macsz/Projects/tf-gam'
FRAMES_PATH = join(ROOT_PATH, 'frames')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-coords-static", type=str, default='0:0x7:7',
                        help="String with face's static frame coordinates "
                             "in a format X1:Y1xX2:Y2")
    args = parser.parse_args()

    params = {
        'face_coords_static': tools.convert_coords(args.face_coords_static),
    }

    print('Static frame\'s cells count:',
          (params['face_coords_static']['x2']-params['face_coords_static'][
              'x1'] + 1) *
          (params['face_coords_static']['y2']-params['face_coords_static'][
              'y1'] + 1))

    files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH)
                    if isfile(join(FRAMES_PATH, f))])
    # co 10ty plik
    # files = files[0::10]
    # n pierwszych
    files = files[:20]

    tools.draw_grid(img=None, open_path=files[0],
                    save_path=join(ROOT_PATH, 'grid.jpg'),
                    face_coords_static=params['face_coords_static'])

    face_classifier = ClassifyFace(files=files, params=params)
    face_classifier.run()

    noses = face_classifier.noses

    nose_classifier = ClassifyNose(noses=noses)
    nose_classifier.run()
