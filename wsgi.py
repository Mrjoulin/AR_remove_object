import logging
from frontend.routes import app

logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)


def app_run(host=None, port=None, debug=None):
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    app.run()
