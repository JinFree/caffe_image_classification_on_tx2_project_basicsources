import caffe
import cv2
import numpy as np


def open_caffe_files():


def frameProcessing(frame):
    output = np.copy(frame)
    return output

def gstreamer_camera_string(camera_num):
    command = "v4l2src device=/dev/video" + str(camera_num) + " ! video/x-raw, width=640, height=480,format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! appsink"
    return command

def video_function(video_input_info, savepath=False):
    cap = cv2.VideoCapture(video_input_info)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = False
    if savepath is not False:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            output = cl.frameProcessing(frame)
            if savepath is True:
                # Write frame-by-frame
                out.write(output)
            # Display the resulting frame
            cv2.imshow("Input", frame)
            cv2.imshow("Output", output)
        else:
            break
        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    if savepath is True:
        out.release()
    cv2.destroyAllWindows()
    return
