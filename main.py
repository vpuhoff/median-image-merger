import os

import cv2
import numpy as np
from numba import jit
from tqdm import tqdm

# Определить путь к папке с изображениями
folder_path = 'P:\stable-diffusion-webui\\outputs\\txt2img-images\\experiments\\median'

# Создать список файлов изображений в папке
image_files = os.listdir(folder_path)

# Рассчитать медианные значения пикселей по каждому каналу (RGB) для каждого пикселя на итоговом изображении
@jit()
def median_pixel_values(images):
    median_img = [[[0,0,0] for _ in range(images[0].shape[1])] for _ in range(images[0].shape[0])]
    for x in range(images[0].shape[0]):
        for y in range(images[0].shape[1]):
            median_b = sorted([images[i][x,y,0] for i in range(len(images))])[len(images)//2]
            median_g = sorted([images[i][x,y,1] for i in range(len(images))])[len(images)//2]
            median_r = sorted([images[i][x,y,2] for i in range(len(images))])[len(images)//2]
            median_img[x][y][0] = median_b
            median_img[x][y][1] = median_g
            median_img[x][y][2] = median_r
    return median_img


# Рассчитать медианные значения пикселей по каждому каналу (RGB) для каждого пикселя на итоговом изображении
@jit()
def average_pixel_values(images):
    median_img = [[[0,0,0] for _ in range(images[0].shape[1])] for _ in range(images[0].shape[0])]
    for x in range(images[0].shape[0]):
        for y in range(images[0].shape[1]):
            median_b = round(sum([images[i][x,y,0] for i in range(len(images))])/ len(images), 0)
            median_g = round(sum([images[i][x,y,1] for i in range(len(images))])/ len(images), 0)
            median_r = round(sum([images[i][x,y,2] for i in range(len(images))])/ len(images), 0)
            median_img[x][y][0] = median_b
            median_img[x][y][1] = median_g
            median_img[x][y][2] = median_r
    return median_img


# Загрузить изображения в список
images = []
for file in tqdm(image_files):
    img = cv2.imread(os.path.join(folder_path, file))  # type: ignore
    images.append(img)

# Преобразовать список изображений в трехмерный массив NumPy для удобства работы с пикселями
images_array = np.array(images)

# Рассчитать медианные значения пикселей по каждому каналу (RGB)
print("Step 1. Calculate median")
median_img = median_pixel_values(images_array)
median_img = np.array(median_img)
cv2.imwrite('median_image.png', median_img)

print("Step 2. Calculate average")
average_img = average_pixel_values(images_array)
average_img = np.array(average_img)
cv2.imwrite('average_image.png', average_img)

print("Step 3. Merge")
images = [median_img, average_img]
images_array = np.array(images)
merged_img = average_pixel_values(images_array)
merged_img = np.array(merged_img)
cv2.imwrite('merged_image.png', merged_img)

