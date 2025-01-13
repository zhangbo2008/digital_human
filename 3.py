# dlib可以实现识别人脸,并且给出识别box的得分!

# 有的图片没识别到人脸. 看看
'''
./db/test1/000168.jpg,这个图片没识别到人脸,坐标:(474, 1075, 690, 1363)
No face dlib landmarks detected for this frame.
./db/test1/000238.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000239.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000240.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000244.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000245.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000247.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000248.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000249.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000251.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000252.jpg,这个图片没识别到人脸,坐标:(474, 1052, 732, 1396)
No face dlib landmarks detected for this frame.
./db/test1/000267.jpg,这个图片没识别到人脸,坐标:(450, 1052, 708, 1396)


'''


import cv2
full_frame=cv2.imread('./db/test1/000168.jpg')
import dlib
detector = dlib.get_frontal_face_detector()
gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 0)
print(faces)
# rectangles[[(448, 567) (663, 782)], [(492, 1129) (672, 1309)]]


# http://dlib.net/   dlib的官网. 

# py api



#       pip install numpy

import sys

import dlib

detector = dlib.get_frontal_face_detector()

pics=['./db/test1/000168.jpg']
for f in pics:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))





# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations. 最后一个参数用来判断朝向.
if 1:
    img = dlib.load_rgb_image(pics[0])
    dets, scores, idx = detector.run(img, 1, 0)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))
