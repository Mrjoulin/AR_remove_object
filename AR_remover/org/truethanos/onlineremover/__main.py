from kivy.app import App  # for the main app
from kivy.uix.floatlayout import FloatLayout  # the UI layout
from kivy.uix.label import Label  # a label to show information
from plyer import camera  # object to read the camera


class UI(FloatLayout):  # the app ui
    def __init__(self, **kwargs):
        super(UI, self).__init__(**kwargs)
        self.lblCam = Label(text="Click to take a picture!")  # create a label at the center
        self.add_widget(self.lblCam)  # add the label at the screen

    def on_touch_down(self, e):
        camera.take_picture('/storage/sdcard0/example.jpg',
                            self.done)  # Take a picture and save at this location. After will call done() callback

    def done(self, e):  # receive e as the image location
        self.lblCam.text = e  # update the label to the image location


class Camera(App):  # our app
    def build(self):
        ui = UI()  # create the UI
        return ui  # show it

    def on_pause(self):
        # when the app open the camera, it will need to pause this script. So we need to enable the pause mode with this method
        return True

    def on_resume(self):
        # after close the camera, we need to resume our app.
        pass


Camera().run()  # start our app