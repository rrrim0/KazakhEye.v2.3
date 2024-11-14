from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import json
import pickle  # Для сохранения и загрузки данных
from sklearn.model_selection import train_test_split
import string
from keras.utils import img_to_array

# Алфавит
alphabet = string.ascii_uppercase + string.digits
char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Словарь регионов Казахстана
regions = {
    "01": "г. Нур-Султан (Астана)",
    "02": "г. Алматы",
    "03": "Акмолинская область",
    "04": "Актюбинская область",
    "05": "Алматинская область",
    "06": "Атырауская область",
    "07": "Западно-Казахстанская область",
    "08": "Жамбылская область",
    "09": "Карагандинская область",
    "10": "Костанайская область",
    "11": "Кызылординская область",
    "12": "Мангистауская область",
    "13": "Туркестанская область",
    "14": "Павлодарская область",
    "15": "Северо-Казахстанская область",
    "16": "Восточно-Казахстанская область",
    "17": "г. Шымкент",
    "18": "Абайская область",
    "19": "Жетысуская область",
    "20": "Улытауская область",
}

# Параметры
input_shape = (64, 128, 1)  # Размер изображения
num_classes = len(alphabet)  # Количество символов
max_seq_len = 8  # Максимальная длина номера

# Пути к данным
image_dir = './data/train/img'
annotations_dir = './data/train/ann'
processed_data_path = './processed_data.pkl'  # Путь к обработанным данным

# Функция для создания модели
def build_model(input_shape, num_classes, max_seq_len):
    model = models.Sequential()

    # CNN-слои
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Извлечение признаков
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Исправленный Reshape
    model.add(layers.RepeatVector(max_seq_len))  # Создаем временные шаги
    model.add(layers.LSTM(128, return_sequences=True))

    # Выходной слой
    model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Функция для загрузки изображений и меток
def load_images_and_labels(image_dir, annotations_dir):
    print("[INFO] Загружаем изображения и аннотации...")
    images = []
    labels = []

    for image_filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_filename)
        annotation_path = os.path.join(annotations_dir, os.path.splitext(image_filename)[0] + '.json')

        # Загружаем изображение
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 64))
        img = img_to_array(img) / 255.0

        # Загружаем аннотацию
        with open(annotation_path) as f:
            annotation = json.load(f)
        label = annotation['description']

        images.append(img)
        labels.append(label)

    print("[INFO] Данные успешно загружены.")
    return np.array(images), labels

# Кодировка меток
def encode_labels(labels, max_seq_len):
    print("[INFO] Кодируем метки...")
    encoded_labels = []
    for label in labels:
        encoded = [char_to_idx.get(char, 0) for char in label]
        encoded += [0] * (max_seq_len - len(encoded))  # Дополнение
        encoded_labels.append(encoded[:max_seq_len])
    print("[INFO] Метки успешно закодированы.")
    return np.array(encoded_labels)

# Проверка наличия обработанных данных
if os.path.exists(processed_data_path):
    print("[INFO] Обнаружены предварительно обработанные данные. Загружаем...")
    with open(processed_data_path, 'rb') as f:
        images, labels = pickle.load(f)
    print("[INFO] Данные успешно загружены.")
else:
    print("[INFO] Обработанные данные отсутствуют. Выполняем загрузку и обработку...")
    images, labels = load_images_and_labels(image_dir, annotations_dir)
    encoded_labels = encode_labels(labels, max_seq_len)
    print("[INFO] Сохраняем обработанные данные для будущего использования...")
    with open(processed_data_path, 'wb') as f:
        pickle.dump((images, encoded_labels), f)
    print("[INFO] Обработанные данные успешно сохранены.")

# Разделение на обучающую и тестовую выборки
print("[INFO] Разделяем данные на обучающую и тестовую выборки...")
encoded_labels = encode_labels(labels, max_seq_len)
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
print("[INFO] Данные успешно разделены.")

# Строим модель
model = build_model(input_shape, num_classes, max_seq_len)
model.summary()

# Обучение модели
print("[INFO] Начало обучения модели...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
print("[INFO] Обучение завершено.")

# Оценка модели
print("[INFO] Оцениваем модель...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"[INFO] Test loss: {loss}")
print(f"[INFO] Test accuracy: {accuracy}")

# Функция для обработки предсказанного номера
def process_predicted_number(predicted):
    # Удаляем префикс "kz" или "KZ", если он есть
    if predicted.lower().startswith("kz"):
        predicted = predicted[2:]

    # Проверяем длину предсказанного номера
    if len(predicted) < 8:
        return f"Предсказанный номер: {predicted}\nРегион: Регион не обнаружен"

    # Извлекаем последние две цифры номера
    region_code = predicted[-2:]

    # Проверяем код региона в таблице
    region_name = regions.get(region_code, "Регион не обнаружен")

    # Возвращаем результат
    return f"Предсказанный номер: {predicted}\nРегион: {region_code} - {region_name}"

# Функция предсказания
def predict_license_plate(model, image_path):
    print(f"[INFO] Выполняем предсказание для {image_path}...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_text = ''.join([idx_to_char[np.argmax(pred)] for pred in prediction[0]])
    print(f"[INFO] Предсказание завершено: {predicted_text}")
    return predicted_text

# Пример использования
predicted_text = predict_license_plate(model, './data/test/img/12190610.jpg-0.png')
print(f"Predicted license plate: {predicted_text}")
