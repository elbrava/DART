import random

import cv2
import numpy as np
from fer import FER
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
detector = FER()
print(detector)
webcam = cv2.VideoCapture(0)
while True:
    sucessful_frame_read, frame = webcam.read()
    img_Aug = frame.copy()
    results = detector.detect_emotions(frame)
    print(list(results).__len__())
    # bounding_box = result[0]["box"]
    # emotions = result[0]["emotions"]
    for result in list(results):
        bounding_box = result["box"]
        print(bounding_box)
        color = (random.randrange(256), random.randrange(256), random.randrange(256))
        emotions = result["emotions"]
        # print(emotions)
        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (
            bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      color, 2, )
        emotion_name, score = detector.top_emotion(frame)
        if score is not None:
            val = f"{score * 100}%"
        else:
            val = f"{score}"
        mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        cv2.imshow("pre", img_Aug)
        cv2.putText(frame, f"{emotion_name}:{val}",
                    (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + 2 * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA, )
        cv2.rectangle(mask, (bounding_box[0], bounding_box[1]), (
            bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      color, -1, )
        mask_inv = cv2.bitwise_not(mask)
        img_Aug = cv2.bitwise_and(img_Aug, img_Aug, mask=mask)
        img_Aug = frame - img_Aug
        x, y, w, h = bounding_box
        img_Aug[y:y + h, x:x + w ] = cv2.resize(frame, (w, h))

        cv2.imshow("webcam", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("sh", mask_inv)
        cv2.imshow("dd", img_Aug)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
