import asyncio
import logging
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from aiogram import Bot, Dispatcher, types, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.fsm.storage.memory import MemoryStorage

import config
from constants import messages_text

LOG_FILE_NAME = 'log/qr_code_checking.log'
device = torch.device('cpu')
IMG_SIZE = 512
NUM_SEED = 41

data_transform_test = A.Compose(
    [
        A.Resize(width=IMG_SIZE, height=IMG_SIZE, p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.Normalize((0.5,), (0.5,))
])
class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


bot = Bot(token=config.READER_BOT_TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Отправь мне изображение счётчика, и я скажу, какие цифры на нём.")

@dp.message(F.photo)
async def download_photo(message: types.Message):
    path = f"tmp/{message.photo[-1].file_id}.jpg"
    await bot.download(
        message.photo[-1],
        destination=path
    )
    # result = 'цифры не найдены'

    result = get_digits(path)
    await message.reply(result)
async def main():
    await dp.start_polling(bot)

def crop_to_mask(image, mask):
    BORDER = 5

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Предполагаем, что маска имеет один контур
    if len(contours) == 0:
      return image

    contour = contours[0]
    rect = cv2.minAreaRect(contour)

    width = int(rect[1][0]) + BORDER
    height = int(rect[1][1]) + BORDER
    if width < height:
        width, height = height, width
        angle = rect[2] - 90  # скорректируйте угол на 90 градусов, если необходимо
    else:
        angle = rect[2]
    rect = (rect[0], (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # Обрезать изображение
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # print(width, height)
    # Примените перспективное преобразование
    cropped_image = cv2.warpPerspective(image, M, (width, height))

    return cropped_image

def calculate_brightness(image, strip_width=5):

    # Проверка, цветное ли изображение, и конвертация в градации серого, если необходимо
    if len(image.shape) == 3:  # Если изображение имеет три канала (цветное)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Если изображение уже в градациях серого

    # Получение размеров изображения
    height, width = gray_image.shape

    # Список для хранения средней яркости каждой полосы
    brightness_values = []

    # Проход по каждой вертикальной полосе шириной 2 пикселя
    for x in range(0, width, strip_width):
        # Убедиться, что не выходим за границы изображения
        if x + strip_width > width:
            strip = gray_image[:, x:width]
        else:
            strip = gray_image[:, x:x+strip_width]

        # Вычисление средней яркости полосы
        mean_brightness = np.mean(strip)
        brightness_values.append(mean_brightness)

    return brightness_values

def smooth_values(values, window_size=4):
    smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    return smoothed_values

def plot_brightness(brightness_values, strip_width=5):
    # Сглаживание значений яркости
    smoothed_values = smooth_values(brightness_values)
    peaks, _ = find_peaks(smoothed_values)
    return len(peaks)

def split_image(image, num_splits):
    height, width, _ = image.shape

    # Вычисление ширины каждого кусочка
    split_width = width // num_splits

    # Список для хранения кусков изображения
    image_splits = []

    for i in range(num_splits):
        if i == num_splits - 1:  # Последний кусочек может быть шире из-за деления
            split = image[:, i*split_width:]
        else:
            split = image[:, i*split_width:(i+1)*split_width]
        image_splits.append(split)

    return image_splits
def get_digits(path):
    modelseg = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )
    modelseg = nn.DataParallel(modelseg)
    modelx = modelseg
    modelx.to(device)
    modelx.load_state_dict(torch.load('model-003a.pth', map_location=torch.device(device)))
    model = ClassificationCNN().to(device)
    model.load_state_dict(torch.load('digit_classifier.pth', map_location=torch.device(device)))


    test_image = cv2.imread(path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    imgsrc_resize = cv2.resize(test_image, [IMG_SIZE, IMG_SIZE])

    with torch.no_grad():
        aug = data_transform_test(image=test_image)
        aug = aug['image'].unsqueeze(dim=0)
        outputs = modelx(aug.cpu())
        preds = outputs.squeeze(dim=[0, 1]).detach().cpu()
        preds = torch.where(preds > 0.5, 1, 0)
        preds = preds.numpy().astype('uint8')

    imgsrc_resize = crop_to_mask(imgsrc_resize, preds)

    brightness_values = calculate_brightness(imgsrc_resize)
    split_count = 5 if plot_brightness(brightness_values) < 5 else 8
    digits = split_image(imgsrc_resize, split_count)

    result = ''
    with torch.no_grad():
        for i in range(split_count):
            image = digits[i]
            image = transform(image).cpu()
            outputs = model(image)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            result += str(int(predicted[0]))
            print(str(predicted[0]) + str(_))
    return  result

if __name__ == "__main__":
    try:
        log_file = open(LOG_FILE_NAME)
    except:
        log_file = open(LOG_FILE_NAME, 'w')
    logging.basicConfig(filename=LOG_FILE_NAME, level=logging.INFO)
    asyncio.run(main())
