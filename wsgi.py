import logging
import argparse
import absl.logging
from server.routes import run_app
from server.bots.run import run_bots

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logging.basicConfig(
    format='[%(filename)s:%(lineno)s - %(funcName)20s()]%(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Server options')
    parser.add_argument("--host", default=None, help="Host server")
    parser.add_argument("--port", type=int, default=5000, help="Port server (default: 5000)")
    parser.add_argument("--use-ssl", action='store_true', default=False, help='Use SSL certificate (for HTTPS)')
    args = parser.parse_args()

    # Ran bots
    logging.info('Start bots')
    run_bots()

    run_app(port=args.port, host=args.host, use_cert=args.use_ssl)
