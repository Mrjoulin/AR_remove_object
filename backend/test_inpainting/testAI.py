import requests
from urllib.request import urlretrieve
from PIL import Image


def get_img(input_file_path, output_file_path):
    output = requests.post(
        "https://api.deepai.org/api/inpainting",
        files={
            'image': open(input_file_path, 'rb'),
        },
        headers={'api-key': '480b520b-21bf-4d3c-9c92-6e84707ee39e'}
    ).json()
    print(output)

    urlretrieve(output['output_url'], output_file_path)


def refactor_img(init_img_path, input_file_path, output_file_path):
    img = Image.open(input_file_path)
    init_img = Image.open(init_img_path)
    crop_img = img.crop((
        128,
        0,
        256,
        128
    ))
    resize_img = crop_img.resize((init_img.width, init_img.height), Image.ANTIALIAS)
    resize_img.save(output_file_path)
    resize_img.show()


if __name__ == '__main__':
    init_img = '/home/joulin/projects/neuralNetwork/src/hackaton/delete_hand/grey_1.jpg'
    output_img = './test_inpainting/output.png'
    output_refactor_img = './test_inpainting/output_refactor.jpg'

    get_img(init_img, output_img)
    refactor_img(init_img, output_img, output_refactor_img)
