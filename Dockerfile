### ----- INSTALL TENSORRT ----- ###
FROM tensorflow/tensorflow:1.14.0-gpu-py3

### ----- INSTALL TENSORRT OPEN SOURCE SOFTWARE ----- ###
# Install required libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip

# Install Cmake
RUN cd /tmp &&\
   wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh &&\
   chmod +x cmake-3.14.4-Linux-x86_64.sh &&\
   ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license &&\
   rm ./cmake-3.14.4-Linux-x86_64.sh

### ----- INSTALL PACKAGES FOR RUNNING WEBCAM INFERENCE ----- ###

# Install OpenCV
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libsm6 libxext6 libxrender-dev python3-tk
RUN pip3 install opencv-python

# Export environment variable for webcam functionality
ENV QT_X11_NO_MITSHM=1

### PROJECT REQUIREMENTS

ADD requirements.txt .
RUN pip3 install setuptools
RUN apt-get install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config
RUN pip3 install -r requirements.txt

# Install tensorflow models object detection
RUN mkdir -p /tensorflow && cd /tensorflow && git clone https://github.com/tensorflow/models
RUN apt-get install -y protobuf-compiler python-pil python-lxml python-tk
RUN cd /tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && apt-get update &&\
   pip3 install . && cd slim && pip3 install . && cd ..
ENV PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

ADD . .
EXPOSE 5000

# Return to project directory and open a terminal
WORKDIR /mnt
CMD /bin/bash
