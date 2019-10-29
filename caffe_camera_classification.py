import caffe
import cv2
import numpy as np


def open_caffe_files():
    caffe.set_mode_gpu()
    ###
    ## ImageNet AlexNet Classification Demo
    #ARCHITECTURE = './AlexNet/alexnet_deploy.prototxt'
    #WEIGHTS = './AlexNet/bvlc_alexnet.caffemodel'
    #MEAN_IMAGE = './AlexNet/ilsvrc_2012_mean.npy'
    #LABEL_FILE = './AlexNet/imagenet_2012_classes.txt'
    #dst_size = (227, 227)
    #mean_image = np.load(MEAN_IMAGE)
    #mean_image = mean_image.mean(1).mean(1)
    
    ## Thresh AlexNet Classification
    ARCHITECTURE = './model/deploy.prototxt'
    WEIGHTS = './model/model.caffemodel'
    MEAN_IMAGE = './model/mean.jpg'
    LABEL_FILE = './model/classes.txt'
    dst_size = (256, 256)
    mean_image = cv2.imread(MEAN_IMAGE)
    mean_image = mean_image.mean(0).mean(0)
    ###
    net = caffe.Classifier(ARCHITECTURE,
                           WEIGHTS,
                           image_dims=dst_size)

    labels = np.loadtxt(LABEL_FILE, str, delimiter='\t')
    return net, mean_image, labels, dst_size

def frameProcessing(frame, net, mean_image, labels, dst_size):
    output = np.copy(frame)
    ready_image = cv2.resize(frame, dst_size)
    ready_image = ready_image - mean_image
    prediction = net.predict([ready_image])
    image_class = labels[prediction.argmax()]
    print(image_class, prediction.argmax())
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
    net, mean_image, labels, dst_size = open_caffe_files()
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
            output = frameProcessing(frame, net, mean_image, labels, dst_size)
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

if __name__ == '__main__':
    camera = gstreamer_camera_string(1)
    video_function(camera)
