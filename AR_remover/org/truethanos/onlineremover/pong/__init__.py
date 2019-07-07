from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.clock import Clock

from time import time

class ListScreen(Screen):

    items_box = ObjectProperty(None)

    def on_enter(self):
        start = time()
        for i in range(0, 50):
            self.items_box.add_widget(ListItem('Item '+str(i)))
        self.items_box.bind(minimum_height=self.items_box.setter('height'))
        print(time()-start)

    def on_leave(self):
        self.items_box.clear_widgets()

class ListItem(BoxLayout):

    title = StringProperty('')

    def __init__(self, title, **kwargs):
        super(ListItem, self).__init__(**kwargs)
        self.title = title

class ListApp(App):

    sm = ScreenManager()
    screens = {}

    def build(self):
        self.__create_screens()
        ListApp.sm.add_widget(ListApp.screens['list1'])
        Clock.schedule_interval(self._switch, 1)
        return ListApp.sm

    def _switch(self, *args):
        ListApp.sm.switch_to(ListApp.screens['list1' if ListApp.sm.current != 'list1' else 'list2'])

    def __create_screens(self):
        ListApp.screens['list1'] = ListScreen(name='list1')
        ListApp.screens['list2'] = ListScreen(name='list2')

if __name__ == '__main__':
    ListApp().run()
