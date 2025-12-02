# Классификация видов птиц (Bird Species Classification)
Проектный практикум МИФИ · Мониторинг экосистем с помощью IT-решений

## 1. Описание проекта
Проект представляет собой прототип IT-решения для мониторинга экосистем, основанный на автоматической классификации видов птиц по изображениям. Используется предобученная модель компьютерного зрения, способная определять вид птицы по одной фотографии без дообучения.

## 2. Проблема и мотивация
Биоразнообразие — ключевой индикатор состояния природных систем. Мониторинг видов птиц позволяет:
- отслеживать изменения ареалов;
- фиксировать редкие и исчезающие виды;
- анализировать влияние климатических и антропогенных факторов.

Ручной анализ изображений плохо масштабируется, поэтому автоматическая классификация:
- ускоряет обработку больших объёмов данных,
- снижает долю ручного труда,
- повышает воспроизводимость и объективность результатов.

## 3. Выбранный аспект экосистемы
Автоматическая идентификация видов птиц по одному изображению.
Классификация выполняется предобученной моделью из HuggingFace, работающей в zero-shot режиме (без дообучения на нашем датасете).

## 4. Используемая модель
- **Модель:** `chriamue/bird-species-classifier`
- **Источник:** HuggingFace Hub  
- **Задача:** Image Classification  
- **Особенность:** Zero-shot (используем как есть)

Пример использования модели:

    from transformers import pipeline

    pipe = pipeline(
        task="image-classification",
        model="chriamue/bird-species-classifier"
    )

    result = pipe("path_or_url_to_image.jpg")
    print(result)

## 5. Используемый датасет
Для тестирования модели используется открытый датасет:

- **Название:** `stealthtechnologies/birds-images-dataset`
- **Источник:** Kaggle
- **Содержимое:** изображения птиц различных видов
- **Назначение:** проверка качества предсказаний модели на реальных данных

Пример загрузки датасета с помощью `kagglehub`:

    import kagglehub
    import os

    path = kagglehub.dataset_download("stealthtechnologies/birds-images-dataset")

    for root, dirs, files in os.walk(path):
        print("Dirs:", dirs)
        print("Files:", files[:5])
        break

## 6. Архитектура решения
1. Загрузка тестового набора изображений из Kaggle.
2. Инициализация предобученной модели с HuggingFace.
3. Прогон изображений через модель.
4. Получение топ-5 предсказаний для каждого изображения.
5. Анализ и интерпретация результатов классификации.

## 7. Запуск проекта в Google Colab

### 7.1. Открытие проекта
1. Открыть Google Colab.
2. Перейти во вкладку **GitHub**.
3. Найти репозиторий `YashinSergey/bird-classification-ml`.
4. Открыть ноутбук `bird-classification.ipynb`.

Альтернатива через `git clone` в ячейке Colab:

    !git clone https://github.com/YashinSergey/bird-classification-ml.git
    %cd bird-classification-ml

### 7.2. Установка зависимостей

В первой кодовой ячейке выполнить:

    !pip install transformers
    !pip install torch
    !pip install Pillow
    !pip install kagglehub
    !pip install numpy

### 7.3. Импорт библиотек

    from transformers import pipeline
    import kagglehub
    import numpy as np
    from PIL import Image
    import os

### 7.4. Запуск классификации на тестовом изображении

    pipe = pipeline("image-classification", model="chriamue/bird-species-classifier")

    result = pipe(
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/hub/parrots.png"
    )
    print(result)

### 7.5. Запуск классификации на изображениях из Kaggle-датасета

    import kagglehub
    import os

    path = kagglehub.dataset_download("stealthtechnologies/birds-images-dataset")

    for root, dirs, files in os.walk(path):
        sample_files = files[:5]
        break

    for fname in sample_files:
        img_path = os.path.join(path, fname)
        preds = pipe(img_path)
        print(fname, "->", preds[:3])

## 8. Примеры результатов

Пример вывода модели для одного изображения:

    [
      {"label": "AMERICAN ROBIN", "score": 0.86},
      {"label": "RED BROWED FINCH", "score": 0.03},
      {"label": "PAINTED BUNTING", "score": 0.02}
    ]

Другие примеры:
- Сова → `GREAT GRAY OWL` (score ≈ 0.60)
- Лебедь → `TRUMPETER SWAN` (score ≈ 0.61)

## 9. Ограничения и возможное развитие

### Ограничения
- Модель не обучалась специально на Kaggle-датасете.
- Не все виды из датасета могут присутствовать в классовом словаре модели.
- Возможны ошибки на визуально схожих видах.

### Возможные направления развития
- Дообучение модели на расширенном датасете по конкретному региону.
- Создание веб-интерфейса (Flask / FastAPI / Streamlit) для загрузки изображений.
- Интеграция геоданных и временных меток для пространственно-временного анализа.
- Сравнение нескольких моделей (ResNet / ViT / ConvNeXt и др.) по качеству и скорости.

## 10. Автор
**Sergey Yashin**  
GitHub: https://github.com/YashinSergey

