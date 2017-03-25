import tools
from classify_face import ClassifyFace
from classify_nose import ClassifyNose
from os import listdir
from os.path import isfile, join

FRAMES_PATH = '/home/macsz/Projects/tf-gam/frames'

if __name__ == '__main__':
    # co 10ty plik
    # files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH) if isfile(join(FRAMES_PATH, f))])[0::10]
    # 5 pierwszych
    files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH) if isfile(join(FRAMES_PATH, f))])[:40]
    # wszystkie
    # files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH) if isfile(join(FRAMES_PATH, f))])

    tools.draw_grid(img=None, open_path=files[0], save_path='/home/macsz/Projects/deep_learning/grid.jpg')

    face_classifier = ClassifyFace(files=files)
    face_classifier.run()

    noses = face_classifier.noses

    nose_classifier = ClassifyNose(noses=noses)
    nose_classifier.run()
