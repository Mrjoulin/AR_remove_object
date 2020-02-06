echo "Setting envivonment variables for the webcam" 
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Build the dockerfile for the engironment 
if [ ! -z $(docker images -q blurred:latest) ]; then
	echo "Dockerfile has already been built"
else
	echo "Building docker image" 
	docker build -f Docker/Dockerfile --tag=blurred .
fi

# Start the docker container
echo "Starting docker container"
if [ $1 == "local" ]; then
  docker run -p 5000:5000 --gpus all -it -v `pwd`:/mnt --device=/dev/video0 -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH blurred
else
  {
  docker run -p 5000:5000 --gpus all -it -v `pwd`:/mnt -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH blurred
  } || {
  apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip
  apt-get install -y libsm6 libxext6 libxrender-dev python3-tk libavdevice-dev libavfilter-dev \
    libopus-dev libvpx-dev pkg-config protobuf-compiler python-pil python-lxml nano
  pip3 install -r Docker/requirements.txt
  mkdir -p /tensorflow && cd /tensorflow && git clone https://github.com/tensorflow/models
  cd /tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && apt-get update &&\
   pip3 install . && cd slim && pip3 install . && cd ..
  PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  }
fi