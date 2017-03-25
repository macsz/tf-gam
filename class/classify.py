import tools
from classify_face import ClassifyFace
from classify_nose import ClassifyNose
from os import listdir
from os.path import isfile, join

ROOT_PATH = '/home/macsz/Projects/tf-gam'
FRAMES_PATH = join(ROOT_PATH, 'frames')

if __name__ == '__main__':
    files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH)
                    if isfile(join(FRAMES_PATH, f))])
    # co 10ty plik
    # files = files[0::10]
    # n pierwszych
    files = files[:20]

    tools.draw_grid(img=None, open_path=files[0],
                    save_path=join(ROOT_PATH, 'grid.jpg'))

    face_classifier = ClassifyFace(files=files)
    face_classifier.run()

    noses = face_classifier.noses

    nose_classifier = ClassifyNose(noses=noses)
    nose_classifier.run()
