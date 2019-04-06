import requests
import logging


def make_api_request(method_name, **kwargs):
    url = 'http://localhost:5000/' + method_name
    response = requests.post(url, json=kwargs).json()
    logging.debug(str(response))


make_api_request('image_corrector', img='str')


def get_api_request(method_name):
    url = 'http://localhost:5000/' + method_name
    response = requests.get(url=url)
    print(response)


get_api_request('image_corrector')
