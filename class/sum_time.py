import sys

if len(sys.argv) == 1:
    print('Error. Missing param: provide path to log file.')
    exit()

path = sys.argv[1]

face_sum_time = 0.0
face_frames = 0
nose_sum_time = 0.0
nose_frames = 0

with open(path) as f:
    for line in f.readlines():
        if 'Nose' in line:
            nose_frames += 1
            nose_sum_time += float(line.split()[-1])
        elif 'Face' in line:
            face_frames += 1
            face_sum_time += float(line.split()[-1])

nose_avg_time = nose_sum_time/nose_frames
face_avg_time = face_sum_time/face_frames

print('Nose frames: {0}, total time: {1}, avg time: {2}, avg fps: {3}'.format(
    nose_frames, nose_sum_time, nose_avg_time, 1/nose_avg_time))
print('Face frames: {0}, total time: {1}, avg time: {2}, avg fps: {3}'.format(
    face_frames, face_sum_time, face_avg_time, 1/face_avg_time))
print('Total frames: {0}, total time: {1}, avg time: {2}, avg fps: {3}'.format(
    face_frames+nose_frames, face_sum_time+nose_sum_time,
    face_avg_time+nose_avg_time, 1/(face_avg_time+nose_avg_time)))
