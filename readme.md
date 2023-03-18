# Описание

Данный скрипт позволяет рассчитать медианные и средние значения пикселей на изображениях и объединить результаты в одно изображение. Скрипт написан на языке Python, используя библиотеки cv2, numpy, numba, tqdm.

# Использование

Для использования скрипта необходимо поместить изображения в папку и указать путь к папке в переменной `folder_path`. Затем запустить скрипт.

# Основные шаги

1. Рассчитать медианные значения пикселей по каждому каналу (RGB) для каждого пикселя на итоговом изображении
2. Рассчитать средние значения пикселей по каждому каналу (RGB) для каждого пикселя на итоговом изображении
3. Объединить результаты из шагов 1 и 2 в одно изображение

# Результаты

Результаты сохраняются в трех отдельных файлах: `median_image.png`, `average_image.png` и `merged_image.png`.