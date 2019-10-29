#!/bin/bash
sudo add-apt-repository universe -y
sudo add-apt-repository multiverse -y
sudo apt-get update
sudo apt install v4l-utils -y
sudo apt-get install gstreamer1.0-tools gstreamer1.0-alsa \
 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
 gstreamer1.0-libav -y
sudo apt-get install libgstreamer1.0-dev \
 libgstreamer-plugins-base1.0-dev \
 libgstreamer-plugins-good1.0-dev \
 libgstreamer-plugins-bad1.0-dev -y
git clone https://github.com/JinFree/caffe.git
cd caffe
./install_caffe_for_tx2.sh
