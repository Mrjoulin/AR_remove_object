echo "Setting envivonment variables for the webcam" 
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Build the dockerfile for the engironment 
if [ ! -z $(docker images -q ar_remove_object_local:latest) ]; then 
	echo "Dockerfile has already been built"
else
	echo "Building docker image" 
	docker build -f Dockerfile --tag=ar_remove_object_local .
fi

# Start the docker container
echo "Starting docker container" 
docker run --runtime=nvidia -it -v `pwd`:/mnt --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH ar_remove_object_local 
