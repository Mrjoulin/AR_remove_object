import logging
import threading
from server.bots import vk


def run_bots():
    logging.info('Start vk-bot')
    threading.Thread(target=vk.handler.run).start()
    # logging.info('Start telegram bot')
    # threading.Thread(target=tg.handler.run).start()
