import os
import cv2
import json
import numpy as np
import string
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Алфавит и цифры для распознавания
alphabet = string.ascii_uppercase + string.digits
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
num_classes = len(alphabet)  # Количество классов (символов в алфавите)
max_seq_len = 8  # Максимальная длина номера
input_shape = (64, 128, 1)  # Размер изображений после предобработки

# Путь к директориям с изображениями и аннотациями
image_dir = './data/train/img'
annotations_dir = './data/train/ann/'

# Функция для преобразования меток в числовые индексы
def encode_labels(labels, max_seq_len):
    encoded_labels = []
    for label in labels:
        encoded_label = [char_to_idx.get(char, 0) for char in label]
        encoded_label = encoded_label[:max_seq_len] + [0] * (max_seq_len - len(encoded_label))
        encoded_labels.append(encoded_label)
    return np.array(encoded_labels)

# Функция для загрузки изображений и аннотаций
def load_images_and_labels(image_dir, annotations_dir):
    images = []
    labels = []

    for image_filename in os.listdir(image_dir):
        image_name = os.path.splitext(image_filename)[0]

        image_path = os.path.join(image_dir, image_filename)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 64))
        img = img_to_array(img) / 255.0
        images.append(img)

        annotation_path = os.path.join(annotations_dir, image_name + '.json')
        with open(annotation_path) as f:
            annotation = json.load(f)

        labels.append(annotation['description'])

    return np.array(images), labels

# Загрузка данных
images, labels = load_images_and_labels(image_dir, annotations_dir)
encoded_labels = encode_labels(labels, max_seq_len)

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Создание модели
def build_model(input_shape, num_classes, max_seq_len):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Reshape((max_seq_len, 128)))  # Формат [batch_size, max_seq_len, features]
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Обучение модели
model = build_model(input_shape, num_classes, max_seq_len)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Сохранение модели
model.save('license_plate_model.h5')
