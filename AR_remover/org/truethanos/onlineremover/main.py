from kivy.app import App
from kivy.lang import Builder
from kivy.graphics import *
from kivy.config import Config
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.video import Video
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.graphics.texture import Texture

# from plyer.platforms.android.camera import AndroidCamera
import base64
import time
import cv2


with open('main.txt') as kivy_file:
    kivy_script = kivy_file.read()
Builder.load_string(kivy_script)


__version__ = '0.0.5.3'


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class MyCamera(Image):
    def __init__(self, capture=None, fps=0, **kwargs):
        super(MyCamera, self).__init__(**kwargs)
        # self.capture = cv2.VideoCapture("/sdcard2/python-apk/2.mp4")
        # print "file path exist :" + str(os.path.exists("/sdcard2/python-apk/1.mkv"))
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        # print str(os.listdir('/sdcard2/'))
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


# Declare both screens
class OnlineCameraScreen(Screen):
    camera = ObjectProperty(None)
    image_path = StringProperty('storage/DCIM/')

    def on_enter(self):
        #self.camera = self.ids['camera']
        #self.camera.state = 'play'
        if not self.camera:
            layout = self.ids['layout']
            #self.camera = Camera(id='camera', index=0, resolution=(1080, 1920))
            #self.camera.state = 'play'
            self.camera = MyCamera(fps=30)
            layout.add_widget(self.camera, index=1)
        # self.camera = MyCamera()

    def take_shot(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        path = self.image_path + "IMG_{}.png".format(timestr)
        self.camera.export_to_png(path)
        print('Take a picture')
        self.on_success_shot(path=path)

    def on_success_shot(self, path):
        # converting saved image to a base64 string:
        VideoCameraScreen.video_path = path
        screen_manager.current = 'video'

    # converting image to a base64, if you want to send it, for example, via POST:
    def image_convert_base64(self):
        with open(self.image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        if not encoded_string:
            encoded_string = ''
        return encoded_string

    def on_leave(self):
       self.camera.state = 'stop'


class VideoCameraScreen(Screen):
    video_path = StringProperty('')

    def on_enter(self):
        try:
            expansions = ['mp4', 'm4v', 'mov', 'avi', 'asf', 'movie', 'jpg', 'png', 'svg']
            vid = self.video_path.split('/')[-1]
            expansion = vid.split('.')[-1].lower()
            if expansion not in expansions:
                raise NameError('Just a Dummy Exception, write your own')
        except Exception as e:
            print("Exception:", e)
        print(self.video_path)

        layout = self.ids['layout']
        self.video = Video(source=self.video_path, state='play')
        layout.add_widget(self.video, index=1)

    def on_leave(self):
        self.ids['layout'].remove_widget(self.video)
        print('Leave video')


class InitScreen(Screen):
    # Config.set('graphics', 'width', '400')
    # Config.set('graphics', 'height', '700')

    def video_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Выберете видео", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()
        
    def load(self, filename):
        VideoCameraScreen.video_path = filename[0]
        self.dismiss_popup()
        screen_manager.current = 'video'


# Create the screen manager
screen_manager = ScreenManager()
screen_manager.add_widget(InitScreen(name='menu'))
screen_manager.add_widget(OnlineCameraScreen(name='online'))
screen_manager.add_widget(VideoCameraScreen(name='video'))


class MainApp(App):
    def build(self):
        return screen_manager


if __name__ == '__main__':
    MainApp().run()
