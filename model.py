import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import cv2

class WorkingDigitRecognizer:
    def __init__(self):
        self.model = None
        self.model_path = 'my_working_model.h5'
    
    def preprocess_image(self, image_path):
        # Открываем изображение
        img = Image.open(image_path).convert('L')
        
        # Увеличиваем контраст
        img = ImageEnhance.Contrast(img).enhance(2.5)
        
        # Изменяем размер
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Конвертируем в массив
        img_array = np.array(img)
        
        # Инвертируем если нужно
        if np.mean(img_array) > 128:
            img_array = 255 - img_array
        
        # Нормализация
        img_array = img_array.astype('float32') / 255.0
        
        # Добавляем размерности
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def find_digits(self, binary_image):
        """Находит все цифры на изображении"""
        # Поиск контуров
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_regions = []
        for contour in contours:
            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Фильтруем слишком маленькие области
            if w > 10 and h > 20 and w * h > 200:
                digit_regions.append((x, y, w, h))
        
        # Сортируем по x координате (слева направо)
        digit_regions.sort(key=lambda region: region[0])
        
        return digit_regions
    
    def extract_digit(self, image, region, padding=10):
        """Извлекает и подготавливает отдельную цифру"""
        x, y, w, h = region
        
        # Добавляем отступы
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Вырезаем цифру
        digit_img = image[y:y+h, x:x+w]
        
        # Изменяем размер до 28x28
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Нормализация
        digit_img = digit_img.astype('float32') / 255.0
        digit_img = np.expand_dims(digit_img, axis=[0, -1])
        
        return digit_img
    
    def train_model(self, force_retrain=False):
        """Обучение с возможностью принудительной перетренировки"""
        
        # Удаляем старую модель если нужно
        if force_retrain and os.path.exists(self.model_path):
            os.remove(self.model_path)
            print("Старая модель удалена")
        
        # Проверяем, есть ли уже обученная модель
        if os.path.exists(self.model_path) and not force_retrain:
            print(f"Загружаем существующую модель из {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            return
        
        print("Начинаем обучение новой модели...")
        
        # Загружаем данные
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Подготовка
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        # СОЗДАЕМ МОДЕЛЬ
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Компиляция
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Обучение
        print("Обучение...")
        history = self.model.fit(
            x_train, y_train,
            batch_size=32,  # ← МОЖЕТЕ МЕНЯТЬ
            epochs=3,       # ← МОЖЕТЕ МЕНЯТЬ
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Сохраняем
        self.model.save(self.model_path)
        print(f"Модель сохранена в {self.model_path}")
        
        # Показываем точность
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Точность на тестовых данных: {test_acc*100:.2f}%")
    
    def recognize(self, image_path):
        """Распознавание цифры"""
        if self.model is None:
            self.train_model()
        
        # Предобработка
        processed = self.preprocess_image(image_path)
        
        # Предсказание
        predictions = self.model.predict(processed, verbose=0)
        digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Показываем результат
        print(f"\n{'='*40}")
        print(f"РЕЗУЛЬТАТ: Цифра {digit}")
        print(f"УВЕРЕННОСТЬ: {confidence:.2f}%")
        print(f"{'='*40}\n")
        
        # Показываем изображение
        plt.figure(figsize=(6, 4))
        plt.imshow(processed[0, :, :, 0], cmap='gray')
        plt.title(f'Распознано: {digit} (уверенность: {confidence:.1f}%)')
        plt.axis('off')
        plt.show()
        
        return digit, confidence

# ============================================
# ПРОСТАЯ ИНТЕРАКТИВНАЯ ПРОГРАММА
# ============================================

def main():
    print("="*50)
    print("РАСПОЗНАВАНИЕ ЦИФР")
    print("="*50)
    
    recognizer = WorkingDigitRecognizer()
    
    while True:
        print("\nМЕНЮ:")
        print("1. Обучить/переобучить модель")
        print("2. Распознать цифру на изображении")
        print("3. Выйти")
        
        choice = input("\nВыберите действие (1-3): ")
        
        if choice == '1':
            retrain = input("Переобучить модель заново? (да/нет): ")
            force = retrain.lower() == 'да'
            recognizer.train_model(force_retrain=force)
            
        elif choice == '2':
            image_path = input("Введите путь к изображению: ")
            if os.path.exists(image_path):
                recognizer.recognize(image_path)
            else:
                print(f"Файл {image_path} не найден!")
                
        elif choice == '3':
            print("До свидания!")
            break
        else:
            print("Неверный выбор!")

# СОЗДАНИЕ ТЕСТОВОГО ИЗОБРАЖЕНИЯ
def create_test_image():
    """Создает простое тестовое изображение с цифрой"""
    from PIL import ImageDraw, ImageFont
    
    img = Image.new('L', (200, 200), color=255)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 150)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 25), "7", fill=0, font=font)
    img.save("test_digit.png")
    print("Создано тестовое изображение: test_digit.png")
    return "test_digit.png"

if __name__ == "__main__":
    # Создаем тестовое изображение
    if not os.path.exists("test_digit.png"):
        create_test_image()
    
    # Запускаем программу
    main()