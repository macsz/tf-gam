from classify_face import ClassifyFace
from classify_nose import ClassifyNose
from os import listdir
from os.path import isfile, join

FRAMES_PATH = '/home/macsz/Projects/deep_learning/frames'

if __name__ == '__main__':
    # TODO nadpisac sorted
    # co 10ty plik
    # files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH) if isfile(join(FRAMES_PATH, f))])[0::10]
    # 5 pierwszych
    files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH) if isfile(join(FRAMES_PATH, f))])
    # wszystkie
    # files = sorted([join(FRAMES_PATH, f) for f in listdir(FRAMES_PATH) if isfile(join(FRAMES_PATH, f))])

    face_classifier = ClassifyFace(files=files, model_path='/home/macsz/Projects/deep_learning/class/face_training/')
    face_classifier.run()

    noses = face_classifier.noses

    nose_classifier = ClassifyNose(noses=noses, model_path='/home/macsz/Projects/deep_learning/class/nose_training/')
    nose_classifier.run()
