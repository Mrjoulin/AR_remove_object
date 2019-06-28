import sys
import sip
import logging
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QSize,  pyqtSlot, QEvent

logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)


class Main(QWidget):
    def __init__(self):
        super().__init__()

        # Background RGB
        self.backgroundRad = 255
        self.backgroundGreen = 255  # 181
        self.backgroundBlue = 255  # 100
        # link clicked
        self.link_clicked = False
        # Start
        logging.info('Start Welcome window')
        self.initUI()

    def initUI(self):
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(self.backgroundRad, self.backgroundGreen, self.backgroundBlue))
        self.setPalette(p)
        logging.info(f'Set background rgb{self.backgroundRad, self.backgroundGreen, self.backgroundBlue}')

        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.welcome_window()

        '''
        background = QLabel()
        pixmap = QPixmap('render_img.png')
        background.setPixmap(pixmap)
        '''

        oImage = QImage("AR_remover/imgs/render_img.png")
        sImage = oImage.scaled(QSize(435, 653))  # resize Image to widgets size
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole
        self.setPalette(palette)
        #self.adjustSize()
        #self.setGeometry(self.frameGeometry())
        self.move(480, 120)
        self.setWindowTitle('True Thanos')
        self.show()

    def welcome_window(self):
        layout = QVBoxLayout()
        layout.setSpacing(300)

        fonts = [
            [QFontDatabase.addApplicationFont('AR_remover/fonts/Montserrat-Medium.ttf'), "Montserrat Medium"],
            [QFontDatabase.addApplicationFont('AR_remover/fonts/Montserrat-Bold.ttf'), "Montserrat Bold"]
        ]
        logging.info(f'Set fonts in app: {fonts}')

        font = QFont()
        font.setFamily("Montserrat Bold")

        title = QPushButton('TRUE THANOS')
        title.setStyleSheet(''' height: 65;
                                font-family: Montserrat;
                                font-style: normal;
                                font-weight: 500;
                                font-size: 28px;
                                line-height: 34px;
                                text-align: center;
                                color: #FFFFFF;
                                margin-top: 5%;
                                margin-bottom: 5%;
                                background: rgba(180, 112, 204, 0.89);
                                border-radius: 25px;
                                ''')

        main_block = QVBoxLayout()
        main_block.setSpacing(40)
        main_text = QLabel('Стирайте только                      \n'
                           'нужные вам объекты                   \n'
                           '           с видео или онлайн          ')
        font.setPointSize(18)
        main_text.setFont(font)
        main_text.setAlignment(Qt.AlignCenter)
        main_text.setStyleSheet("color: white")

        buttons_block = QVBoxLayout()
        buttons_block.setSpacing(20)
        video_button = QPushButton('Video')
        video_button.setStyleSheet(''' height: 55;
                                       font-family: Montserrat;
                                       font-style: normal;
                                       font-weight: 500;
                                       font-size: 24px;
                                       line-height: 34px;
                                       text-decoration:none;
                                       text-align: center;

                                       background: #13C0CB;
                                       border-radius: 25px;
                                       border: 0;
                                       color: rgba(255, 255, 255, 0.89); 
                                    '''
                                   )
        video_button.clicked.connect(self.welcome_button_click)

        online_button = QPushButton('Online')
        online_button.setStyleSheet('''
                                       height: 55;
                                       font-family: Montserrat;
                                       font-style: normal;
                                       font-weight: 500;
                                       font-size: 24px;
                                       line-height: 34px;
                                       text-align: center;
                                       text-decoration:none;

                                       background: #4286f4;
                                       border-radius: 25px;
                                       border: 0;

                                       color: rgba(255, 255, 255, 0.89);
                                    '''
                                    )
        online_button.clicked.connect(self.welcome_button_click)
        buttons_block.addWidget(video_button)
        buttons_block.addWidget(online_button)

        main_block.addWidget(main_text)
        main_block.addLayout(buttons_block)

        layout.addWidget(title)
        layout.addLayout(main_block)

        if self.layout() is not None:
            self.delete_items_of_layout(self.layout())
            sip.delete(self.layout())

        logging.info('Set layout in welcome window')
        self.setLayout(layout)

    def welcome_button_click(self):
        sender = self.sender()
        logging.info('%s button click' % sender.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    logging.info('Start app')
    ex = Main()

    sys.exit(app.exec_())
