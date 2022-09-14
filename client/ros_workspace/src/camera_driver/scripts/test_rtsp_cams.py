import cv2

vcap1 = cv2.VideoCapture("rtsp://182.168.1.101/stream0")
vcap2 = cv2.VideoCapture("rtsp://182.168.1.102/stream0")
vcap3 = cv2.VideoCapture("rtsp://182.168.1.103/stream0")
vcap3 = cv2.VideoCapture(
    "rtsp://admin:SV123456@182.168.1.103:554/Streaming/Channels/101"
)
vcap4 = cv2.VideoCapture("rtsp://182.168.1.104/stream0")
vcap5 = cv2.VideoCapture("rtsp://182.168.1.105/stream0")
print("CV_CAP_PROP_FPS:  %f\n", vcap1.get(cv2.CAP_PROP_FPS))
print("CV_CAP_PROP_FPS:  %f\n", vcap2.get(cv2.CAP_PROP_FPS))
print("CV_CAP_PROP_FPS:  %f\n", vcap3.get(cv2.CAP_PROP_FPS))
print("CV_CAP_PROP_FPS:  %f\n", vcap4.get(cv2.CAP_PROP_FPS))
print("CV_CAP_PROP_FPS:  %f\n", vcap5.get(cv2.CAP_PROP_FPS))
while 1:

    ret, frame1 = vcap1.read()
    if ret:
        cv2.imwrite("frame_cam_1.jpg", frame1)
        # cv2.namedWindow("VIDEO1", 0)
        # cv2.imshow("VIDEO1", frame1)
        # cv2.waitKey(1)
    else:
        vcap1.release()
        vcap1.open("rtsp://182.168.1.101/stream0")
        print("cannot capture frame for cam_1")

    ret, frame2 = vcap2.read()
    if ret:
        cv2.imwrite("frame_cam_2.jpg", frame2)
        # cv2.namedWindow("VIDEO2", 0)
        # cv2.imshow("VIDEO2", frame2)
        # cv2.waitKey(1)
    else:
        vcap2.release()
        vcap2.open("rtsp://182.168.1.102/stream0")
        print("cannot capture frame for cam_2")

    ret, frame3 = vcap3.read()
    if ret:
        cv2.imwrite("frame_cam_3.jpg", frame3)
        # cv2.namedWindow("VIDEO3", 0)
        # cv2.imshow("VIDEO3", frame3)
        # cv2.waitKey(1)
    else:
        vcap3.release()
        vcap3.open("rtsp://admin:SV123456@182.168.1.154:554/Streaming/Channels/103")
        print("cannot capture frame for cam_3")

    ret, frame4 = vcap4.read()
    if ret:
        cv2.imwrite("frame_cam_4.jpg", frame4)
        # cv2.namedWindow("VIDEO4", 0)
        # cv2.imshow("VIDEO4", frame4)
        # cv2.waitKey(1)
    else:
        vcap4.release()
        vcap4.open("rtsp://182.168.1.104/stream0")
        print("cannot capture frame for cam_4")

    ret, frame5 = vcap5.read()
    if ret:
        cv2.imwrite("frame_cam_5.jpg", frame5)
        # cv2.namedWindow("VIDEO5", 0)
        # cv2.imshow("VIDEO5", frame5)
        # cv2.waitKey(1)
    else:
        vcap5.release()
        vcap5.open("rtsp://182.168.1.105/stream0")
        print("cannot capture frame for cam_5")
