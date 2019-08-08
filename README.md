# AR_remove_object

Have you ever wanted to remove disturbing objects from photos / videos or even online? 

Our application will provide you with this opportunity.

## 1. Clone this project and run the executable file.
```bash
# Clone directory
git clone -b develop git@github.com:Mrjoulin/AR_remove_object.git
cd AR_remove_object
```
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

### WARNING
#### All objects detected by "Tensorflow Object Detection" algoritm.
Therefore, before using the application, make sure that it is installed ([installation guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)).

## 2. Options for use
Our application offers several options for processing directories with video, online, etc.

### 2.1 Option `--masking` and `--inpaint`
These are algorithms for hiding an object by

  ~ generating a uniform pattern (`--masking`)

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
python run.py --render_directory videos/input_videos/to_render/IMG_0080.MOV --masking
```
The final video will be in videos/out_videos directory

### 2.4 Option `--render_image <path-image>`
Images processing using the selected algorithm (see clause 2.1)

Example:
```bash
# Use a --masking algoritm to render an image
python run.py --render_image AR_remover/imgs/smile-people.jpeg --masking
```
The final image will be in AR_remover directory

### 2.5 Option `--render_online`
Use a web camera to online processing objects

Example:
```bash
# Use a web camera for online rendering
python run.py --rendder_online
```
#### Wait kays:
##### `     ~ "  "(Spaсe)` - to masking (press again to turn off)
##### `     ~ "i"` - to impainting (press again to turn off)
##### `     ~ "q"` or Esc - to exit

### 2.6 Option `--tensorflow2`
##### It is appropriate to use the newer *Tensorflow* to process

## 3. API to get a render image
These are requests to our server `xxx.xxx.xxx.xxx:5000` to receive the processed frame.
### 3.1 Init page `/`
Start page with buttons for testing server response
### 3.2 Get masking image `/get_masking_image`
POST request, to obtain an image with an algorithm applied to it `--masking` (see clause 2.1)
#### Input json:
```
    {
       "img": <BASE64-encoded img>,
       "objects": [ {"x": <x>, "y": <y>, "width": <width>, "height": <height>}, ...]
       "class_objects": [<number_class>, ...]
    }
```
#### Output json
```
    {
       "payload": {
          "img": <BASE64-encoded masking image>
    }
    }
```
### 3.3 Get inpaint image `/get_inpaint_image`
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
