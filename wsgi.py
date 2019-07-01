import logging
import argparse
from frontend.routes import app
from frontend.routes_web import run_app

logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Server options')
    parser.add_argument("--host", default=None, help="Host server")
    parser.add_argument("--port", type=int, default=None, help="Port server (default: 5000)")
    parser.add_argument("--debug", action='store_true', default=False, help="Debug server")
    args = parser.parse_args()

    #app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    run_app(port=args.port, host=args.host)
