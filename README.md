# AR_remove_object

Have you ever wanted to remove disturbing objects from photos / videos or even online? 

Our application will provide you with this opportunity.

## 1. Clone this project and run the executable file.
#### 1.1 Clone directory
```bash
# Clone directory
git clone -b develop git@github.com:Mrjoulin/AR_remove_object.git
cd AR_remove_object
```
#### 1.2.1 Python Installation 
And then install all dependencies using `python pip`.
```bash
# Installing a newer python (if necessary)
sudo apt-get install python3
# Install all dependensies
pip install -r requirements.txt
```
And run the executable file
```bash
# Run app
python run.py [any options]
```
Run `python run.py --help` to watch all options

#### 1.2.2 Or you can use docker to install

Install a docker image [there](https://cloud.docker.com/u/dergunovdv/repository/docker/dergunovdv/thanosar)

And run the commands:
```shell script
# Build application
docker build . -t thanosar
# Run application
docker run -it -p 5000:5000 thanosar
```

### WARNING
#### All objects detected by "Tensorflow Object Detection" algoritm.
Therefore, before using the application, make sure that it is installed ([installation guide](https://medium.com/@karol_majek/10-simple-steps-to-tensorflow-object-detection-api-aa2e9b96dc94)).

## 2. Options for use
Our application offers several options for processing directories with video, online, etc.

### 2.1 Option `--inpaint`
These are algorithms for hiding an object by

  ~ smearing nearby areas (`--inpaint`)

### 2.2 Option `--render_directory <path-directory>`
Processing all videos in a folder using one of the algorithms (see clause 2.1)

Example:
```bash
# Use a --inpaint algoritm to render directory with videos
python run.py --render_directory videos/input_videos/to_render/ --inpaint  
```
The final videos will be in videos/out_videos directory

### 2.3 Option `--render_video <path-video>`
Vidos processing using the selected algorithm (see clause 2.1)

Example:
```bash
# Use a --masking algoritm to render a video
python run.py --render_directory videos/input_videos/to_render/IMG_0080.MOV --inpaint
```
The final video will be in videos/out_videos directory

### 2.4 Option `--render_image <path-image>`
Images processing using the selected algorithm (see clause 2.1)

Example:
```bash
# Use a --masking algoritm to render an image
python run.py --render_image server/imgs/render_img.jpeg --inpaint
```
The final image will be in AR_remover directory

### 2.5 Option `--render_online`
Use a web camera to online processing objects

Example:
```bash
# Use a web camera for online rendering
python run.py --render_online
```
#### Wait kays:
##### `     ~ "  "(Spaсe)` - to impainting (press again to turn off)
##### `     ~ "q"` or Esc - to exit

### 2.6 Option `--tensorflow2`
##### It is appropriate to use the newer *Tensorflow* to process

## 3. API to get a render image
These are requests to our server `xxx.xxx.xxx.xxx:5000` to receive the processed frame.
### 3.1 Init page `/`
Start page with buttons for testing server response
### 3.2 Get inpaint image `/get_inpaint_image`
POST request, to obtain an image with an algorithm applied to it `--inpaint` (see clause 2.1)
#### Input json:
```
    {
       "img": <BASE64-encoded img>,
       "objects": [ {"x": <x>, "y": <y>, "width": <width>, "height": <height>}, ...]
    }
```
#### Output json
```
    {
       "payload": {
          "img": <BASE64-encoded inpaint image>
       }
    }
```
### 3.3 WebRTC connection `/offer`
POST request to WebRTC connection with server. Rendering a stream video
#### Input json:
```
    {
      "sdp": <string>, 
      "type": <string>, # - for WebRTC connection
      "video_transform": {
                        "name": <name_algorithm> # - Options: "boxes", "inpaint", "edges", "cartoon" or empty "".
                        "src": [<additional variables>] # - for "inpaint" -- [<class objects>] (For example: ["people"]) 
                                                        # - for others -- []
                        }
    }
```
#### Output json:
```
    {
      "sdp": <string>,
      "type": <string> # - for webRTC connection
    }
```



