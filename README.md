# AR_remove_object

Have you ever wanted to remove disturbing objects from photos / videos or even online? 

Our application will provide you with this opportunity.

## 1. Installation
####You can install the project in 3 ways:
#### 1.1.1 Clone Docker image (if you have access)
```shell script
docker pull joulin/blurred:latest
```

#### 1.1.2 Clone and run Docker (recommended)
```bash
# Clone directory
git clone -b develop git@github.com:Mrjoulin/AR_remove_object.git
cd AR_remove_object
sudo sh setup.sh
# And after Docker start and build
```
#### 1.1.3 Python Installation 
And then install all dependencies using `python pip`.
```bash
# Installing a newer python (if necessary)
sudo apt-get install python3
# Install all dependensies
apt-get install -y libsm6 libxext6 libxrender-dev python3-tk libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config
pip install -r Docker/requirements.txt
```
#### WARNING
All objects detected by "Tensorflow Object Detection" algoritm.
Therefore, before using the application, make sure that it is installed ([installation guide](https://medium.com/@karol_majek/10-simple-steps-to-tensorflow-object-detection-api-aa2e9b96dc94)).

### 1.2 And run the executable file
```bash
# if you use Docker, then after Docker start and build, your terminal will open
# To run server
python wsgi.py [any options]
# To run app localy (use webcam)
python run.py [any options]
```
Run `python <name_file>.py --help` to watch all options


## 2. Options for use for `run.py`
Our application offers several options for processing directories with video, online, etc.

### 2.1 Option `--render_directory <path-directory>`
Processing all videos in a folder using one of the algorithms (see clause 2.1)

Example:
```bash
# Use a --inpaint algoritm to render directory with videos
python run.py --render_directory videos/input_videos/to_render/  
```
The final videos will be in videos/out_videos directory

### 2.2 Option `--render_video <path-video>`
Vidos processing using the selected algorithm (see clause 2.1)

Example:
```bash
# Use a --inpaint algoritm to render a video
python run.py --render_video videos/input_videos/to_render/<you_video>.MOV 
```
The final video will be in videos/out_videos directory

### 2.3 Option `--render_image <path-image>`
Images processing using the selected algorithm (see clause 2.1)

Example:
```bash
# Use a --inpaint algoritm to render an image
python run.py --render_image server/imgs/render_img.jpeg
```
The final image will be in AR_remover directory

### 2.4 Option `--render_online`
Use a web camera to online processing objects

Example:
```bash
# Use a web camera for online rendering
python run.py --render_online
```
#### Wait kays:
##### `     ~ "  "(Spaсe)` - to impainting (press again to turn off)
##### `     ~ "q"` or Esc - to exit

## 3. API to get a render image
These are requests to our server `xxx.xxx.xxx.xxx:5000` to receive the processed frame.
### 3.1 Init page `/`
Start page with buttons for testing server response
### 3.2 WebRTC connection `/offer`
POST request to WebRTC connection with server. Rendering a stream video
#### Connection json:
    {
        sdp, type: <string>, <string> - for WebRTC connection
        video_transform: (check Input in Data Channel) - it's need to start
    }
#### Data Channel:
#### Input json:
```
    {
        "message_id": <message id>
        "name": <name_algorithm>, - name algorithm. Options: "", "edges", "boxes", "inpaint"
        "src": [<additional variables>] - for "inpaint" -- [<class id objects>] (For example: [1, 15]; ["all"])
                                          for others -- []
    }
```
#### Output json:
```
    {
        "message_id": <message id>,
        "data": [
            {
                "class_id": <class id object>, - for example: 23 (int)
                "position: {
                    "x_min": <position top left point of the rectangle>, - from the left edge (example: 0,1325..)
                    "y_min": <position top left point of the rectangle>, - from the top edge (example: 0,3271..)
                    "x_max": <position bottom right point of the rectangle>, - from the left edge (example: 0,562..)
                    "y_max": <position bottom right point of the rectangle> - from the top edge (example: 0,8932..)
                }
            },
            ...

        ] - detected objects: for "inpaint" - removed objects
                              for "boxes" - all detected objects
                              for others - []
    }
```
