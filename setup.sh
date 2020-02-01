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
if [ $1 == "server" ]; then
  docker run -p 5000:5000 --gpus all -it -v `pwd`:/mnt -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH blurred
else
  docker run -p 5000:5000 --gpus all -it -v `pwd`:/mnt --device=/dev/video0 -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH blurred
fi