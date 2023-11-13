import cv2
import time
import numpy as np
import imutils


def bgremove(frame, min_thres, min_satur, min_brigth):
    min_piel = np.array([min_thres, min_satur, min_brigth])
    max_piel = np.array([255, 255, 255])

    # Convert image to HSV
    frameHSV = cv2.GaussianBlur(frame, (7, 7), 0)
    frameHSV = cv2.cvtColor(frameHSV, cv2.COLOR_RGB2HSV)

    # mascara que pilla el color del tono de piel
    skinRegion = cv2.inRange(frameHSV, min_piel, max_piel)
    frame_skin = cv2.bitwise_and(frame, frame, mask=skinRegion)

    return frame_skin


def main():
    window_name = "Window"

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('../data/videos/videoChorra3.mp4')


    cv2.namedWindow(window_name)

    # Crating the trackbars
    cv2.createTrackbar("Min", window_name, 0, 255, lambda x: x)  # 90
    cv2.createTrackbar("Min-s", window_name, 0, 255, lambda x: x)  # 60
    cv2.createTrackbar("Min-b", window_name, 0, 255, lambda x: x)  # 50

    while True:
        _, img = cap.read()
        
        img =  imutils.resize(img, width=400)

        masked_img = bgremove(
            img,
            cv2.getTrackbarPos("Min", window_name),
            cv2.getTrackbarPos("Min-b", window_name),
            cv2.getTrackbarPos("Min-s", window_name),
        )

        cv2.imshow(window_name, masked_img)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            name = "skin" + time.asctime() + ".jpg"
            cv2.imwrite(name, masked_img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()