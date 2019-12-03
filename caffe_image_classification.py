import caffe
import cv2
import numpy as np


def open_caffe_files():
    #caffe.set_mode_gpu()
    ###
    ## ImageNet AlexNet Classification Demo
    ARCHITECTURE = './AlexNet/alexnet_deploy.prototxt'
    WEIGHTS = './AlexNet/bvlc_alexnet.caffemodel'
    MEAN_IMAGE = './AlexNet/ilsvrc_2012_mean.npy'
    LABEL_FILE = './AlexNet/imagenet_2012_classes.txt'
    dst_size = (227, 227)
    mean_image = np.load(MEAN_IMAGE)
    mean_image = mean_image.mean(1).mean(1)
    
    ## Thresh AlexNet Classification
    #ARCHITECTURE = './model/deploy.prototxt'
    #WEIGHTS = './model/model.caffemodel'
    #MEAN_IMAGE = './model/mean.jpg'
    #LABEL_FILE = './model/classes.txt'
    #dst_size = (256, 256)
    #mean_image = cv2.imread(MEAN_IMAGE)
    #mean_image = mean_image.mean(0).mean(0)
    ###
    net = caffe.Classifier(ARCHITECTURE,
                           WEIGHTS,
                           image_dims=dst_size)

    labels = np.loadtxt(LABEL_FILE, str, delimiter='\t')
    return net, mean_image, labels, dst_size

if __name__ == '__main__':
    frame = cv2.imread('/home/opencv/OpenCV_in_Ubuntu/Data/sudoku-original.jpg')
    cv2.imshow('image',frame)
    net, mean_image, labels, dst_size = open_caffe_files()
    ready_image = cv2.resize(frame, dst_size)
    ready_image = ready_image - mean_image
    prediction = net.predict([ready_image])
    image_class = labels[prediction.argmax()]
    print(image_class, prediction.argmax(), prediction[0,prediction.argmax()])
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',frame)
    cv2.waitKey()
