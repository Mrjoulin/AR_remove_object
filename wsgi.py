import logging
import argparse
from frontend.routes import run_app

logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Server options')
    parser.add_argument("--host", default=None, help="Host server")
    parser.add_argument("--port", type=int, default=None, help="Port server (default: 8080)")
    args = parser.parse_args()

    run_app(port=args.port, host=args.host)
