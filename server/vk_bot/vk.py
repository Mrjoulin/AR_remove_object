import os
import cv2
import json
import random
import logging
import requests
from botal import Botal

from vk_api import VkApi, VkUpload
from vk_api.longpoll import VkLongPoll
from vk_api.keyboard import VkKeyboard, VkKeyboardColor

# local modules
from server.routes import object_detection
from backend.source import get_image_inpaint, postprocess


VK_TOKEN = 'ae33df9bcec463df92d2759c2fb8cd2419bcfe37945b799cfdabf3806d41af0d3547d028265f8d0c74ee8'
sess = VkApi(token=VK_TOKEN)
api = sess.get_api()
lp = VkLongPoll(sess)


def safe_listen():
    while True:
        try:
            yield from lp.listen()
        except Exception as e:
            logging.exception(e)


handler = Botal(filter(lambda x: x.to_me, safe_listen()), lambda x: x.user_id)


def random_id():
    rand_id = random.randint(1, 999999999999)
    return rand_id


def msg_response(user_id, status_msg='usual'):
    keyboard = VkKeyboard(one_time=True)
    keyboard.add_button('Помощь', color=VkKeyboardColor.PRIMARY)
    if status_msg == 'first':
        text = 'Привет! Отправь мне фотографию для удаления с неё жалких людишек и не только ;-P'
    elif status_msg == 'help':
        text = \
            '''
            Я бот который умеет стирать ненужные тебе объекты (например, людей) с фотографии. Круто правда?
            Прикрепи фотографию и отправь мне, для обработки, всё просто ;-)
            Иногда обработка может занять некоторо время, о не бойся, долго ждать не придётся 👌🏻
            '''
    elif status_msg == 'invalid_attach':
        text = 'Похоже ты прикрепил не тот документ :)\nМне нужна просто фотография/фоторгафии'
    elif status_msg == 'false_feedback':
        text = 'Эх жаль :-(\n Ну ладно, может тебе обработать ещё фотографию?'
    else:
        text = 'Круто! Так что насчёт скинуть мне фото?'

    api.messages.send(user_id=user_id, random_id=random_id(),
                      message=text,
                      keyboard=keyboard.get_keyboard())


def get_photo(message, user_id):
    first_msg = True
    feedback = False
    while True:
        logging.info(f'Get message from user {str(user_id)} : {str.encode(message.text).decode("unicode-escape")}; '
                     f'attachments: {str(message.attachments)}')

        if message.text == 'Отправить фитбэк' and not feedback:
            keyboard = VkKeyboard(one_time=True)
            keyboard.add_button('Отмена', color=VkKeyboardColor.NEGATIVE)
            api.messages.send(user_id=user_id, random_id=random_id(),
                              message='Можешь отправь свой фитбэк мне, или, для отмены, нажми кнопку на клавиатуре.\n'
                                      'Мы будем очень признательны за вашу обратную связь 🙃',
                              keyboard=keyboard.get_keyboard())
            while True:
                message = (yield).text
                if message == 'Отмена':
                    msg_response(user_id, status_msg='false_feedback')
                    message = (yield)
                    break
                elif message == '':
                    api.messages.send(user_id=user_id, random_id=random_id(),
                                      message='Пожалуйста, отправь мне текстовое сообщение для фитбэка, или нажми '
                                              'кнопку на клавиатуре для отмены.',
                                      keyboard=keyboard.get_keyboard())
                else:
                    my_id = 290469615
                    api.messages.send(user_id=my_id, random_id=random_id(),
                                      message=f"Новый фитбэк от @id{str(user_id)}\nТекст сообщения:\n" + message)
                    api.messages.send(user_id=user_id, random_id=random_id(),
                                      message='Спасибо за ваш фитбэк! Теперь вы можете отправить мне ещё фотографию '
                                              'для обработки.')
                    feedback = True
                    message = (yield)
                    break
            continue

        if not message.attachments:
            if message.text == 'Помощь':
                msg_response(user_id, status_msg='help')
            else:
                msg_response(user_id, status_msg=('first' if first_msg else 'usual'))
        else:
            photos = []
            attachments = api.messages.getById(message_ids=message.message_id)['items'][0]['attachments']
            api.messages.send(user_id=user_id, random_id=random_id(),
                              message='Принял на обработку, подожди чуть-чуть 👌🏻')
            for attach in attachments:
                if attach['type'] == 'photo':
                    current_photo = attach['photo']

                    logging.info(f'Photo {str(current_photo["id"])} get from {str(user_id)}')
                    inf = requests.get(current_photo['sizes'][-1]['url'])
                    img_path = 'server/vk_bot/render_imgs/to_render_img_%s.jpg' % current_photo['id']
                    with open(img_path, 'wb') as photo_file:
                        photo_file.write(inf.content)

                    image_np = cv2.imread(img_path)

                    boxes, masks = object_detection(image_np, box=True, mask=True)

                    image_class_id = postprocess(image_np, boxes=boxes, get_class_to_render=True)
                    if image_class_id:

                        labels_file = 'backend/object_detection_labels.json'
                        labels = json.loads(open(labels_file, 'r').read())

                        text_find_labels = "Мне удолось найти на фотографии следующие обыекты:\n"
                        for _id in range(len(image_class_id)):
                            text_find_labels += f'{str(_id + 1)}. ' + \
                                                str.encode(labels[str(image_class_id[_id])]['name']).decode('unicode-escape') + '\n'
                        text_find_labels += 'Напиши мне числа (без запятых и пробелов), под которыми находятся объект' \
                                            'ы, которые ты хочешь убрать, или нажми на одну из кнопок на клавиатуре.'

                        keyboard = VkKeyboard(one_time=True)
                        keyboard.add_button('Стереть всех!', color=VkKeyboardColor.POSITIVE)
                        if 1 in image_class_id:  # people
                            keyboard.add_line()
                            keyboard.add_button('Стереть людей!', color=VkKeyboardColor.NEGATIVE)

                        if len(attachments) > 1:
                            upload = VkUpload(sess)

                            init_photo = upload.photo_messages(photos=img_path)[0]

                            api.messages.send(user_id=user_id, random_id=random_id(),
                                              message=text_find_labels, keyboard=keyboard.get_keyboard(),
                                              attachment='photo{}_{}'.format(init_photo['owner_id'], init_photo['id']))
                        else:
                            api.messages.send(user_id=user_id, random_id=random_id(),
                                              message=text_find_labels, keyboard=keyboard.get_keyboard())

                        message = (yield).text

                        number_classes_to_render = []
                        if message == 'Стереть всех!':
                            number_classes_to_render = image_class_id
                        elif message == 'Стереть людей!':
                            number_classes_to_render.append(1)
                        else:
                            for number_class in message:
                                try:
                                    if int(number_class) in range(1, len(image_class_id) + 1) and \
                                            int(number_class) not in number_classes_to_render:
                                        number_classes_to_render.append(image_class_id[int(number_class) - 1])
                                except ValueError:
                                    pass
                    else:
                        text_not_found = 'Извини, мне не удалось найти ни одного объекта на твоей фотографии :-(\n' \
                                         'Попробуй скинуть другую фотографию'
                        if len(attachments) > 1:
                            upload = VkUpload(sess)

                            init_photo = upload.photo_messages(photos=img_path)[0]

                            api.messages.send(user_id=user_id, random_id=random_id(), message=text_not_found,
                                              attachment='photo{}_{}'.format(init_photo['owner_id'], init_photo['id']))
                        else:
                            api.messages.send(user_id=user_id, random_id=random_id(), message=text_not_found)
                        photos.append(None)
                        continue

                    # Remove input image
                    os.remove(img_path)

                    logging.info('Number classes to render: %s' % number_classes_to_render)
                    final_image_np = get_image_inpaint(image_np, masks=masks, boxes=boxes,
                                                       classes_to_render=number_classes_to_render)
                    img_path = img_path.replace('to_render', 'inpaint')
                    cv2.imwrite(img_path, final_image_np)

                    upload = VkUpload(sess)

                    photo = upload.photo_messages(photos=img_path)[0]

                    photos.append(photo)
                    logging.info(f'Inpaint photo {str(current_photo["id"])} from {str(user_id)} successfully')

                    # Remove generate inpaint photo
                    os.remove(img_path)

            if photos:
                text = 'А вот и ' + ('твоё обработанное' if len(photos) == 1 else 'твои обработанные') + ' фото :-)'
                attachment = ''
                for photo in photos:
                    try:
                        attachment += 'photo{}_{},'.format(photo['owner_id'], photo['id'])
                    except TypeError:
                        pass

                if attachment:
                    api.messages.send(user_id=user_id, random_id=random_id(),
                                      attachment=attachment,
                                      message=text)

                    text = 'Жду ещё твои фотографии для обработки)'
                    if not feedback:
                        text += '\nЕсли не сложно, можешь отправить фитбэк по нашей технологии :)'
                        keyboard = VkKeyboard(one_time=True)
                        keyboard.add_button('Отправить фитбэк', color=VkKeyboardColor.POSITIVE)
                        api.messages.send(user_id=user_id, random_id=random_id(),
                                          message=text, keyboard=keyboard.get_keyboard())
                    else:
                        api.messages.send(user_id=user_id, random_id=random_id(), message=text)
            else:
                msg_response(user_id, status_msg='invalid_attach')

        first_msg = False
        message = (yield)


@handler.handler
def on_message(user_id):
    message = (yield)
    logging.info('Start messaging with user. Id: %s' % user_id)
    yield from get_photo(message,  user_id)


@handler.error_handler(Exception)
def on_error(user_id, e):
    if not isinstance(e, StopIteration):
        logging.exception(e)
        api.messages.send(user_id=user_id, random_id=random_id(),
                          message='Извините, произошла внутренняя ошибка: "{}"'.format(str(e)))


if __name__ == '__main__':
    logging.info('Start bot')
    handler.run()

