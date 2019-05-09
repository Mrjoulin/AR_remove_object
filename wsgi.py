import logging
from backend.routes import app

logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    app.run()
