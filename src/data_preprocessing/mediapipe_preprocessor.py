import mediapipe as mp
import cv2
import numpy as np
from mediapipe import solutions
import random
import torch
from torchvision import transforms, datasets
import os
from collections import defaultdict
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

# Путь к модели MediaPipe
model_path = r'D:\GitHubRepo\ASL-coursework\src\models\hand_landmarker.task'

# Инициализация параметров MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class ASLMPPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_dataset_MPVEC(self, dataset_path, output_path, fraction=0.25):
        # Загружаем датасет БЕЗ transform
        full_dataset = datasets.ImageFolder(dataset_path)

        # Разбиваем на классы
        class_to_samples = defaultdict(list)
        for path, label in full_dataset.samples:
            class_to_samples[label].append((path, label))

        # Отбираем fraction от каждого класса
        reduced_samples = []
        for label, samples in class_to_samples.items():
            quarter_len = max(1, len(samples) * fraction)  # хотя бы 1
            reduced_samples.extend(random.sample(samples, int(quarter_len)))

        random.shuffle(reduced_samples)

        progress_bar = tqdm(
            total=len(reduced_samples),
            desc=f"Обработка изображений ({int(fraction * 100)}% каждой папки)",
            unit="img",
            dynamic_ncols=True
        )

        processed_data = []

        for i, (path, label) in enumerate(reduced_samples):
            image_pil = default_loader(path)  # загружаем изображение вручную
            image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            image_bgr = resize_to_640x480(image_bgr)

            feature_vector = extract_hand_features(image_bgr)

            if feature_vector is not None:
                processed_data.append((feature_vector, label))

            progress_bar.update(1)
            progress_bar.set_postfix({
                "обработано": f"{i + 1}/{len(reduced_samples)}",
                "текущий": f"{full_dataset.classes[label]}"
            })

        progress_bar.close()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        torch.save({
            'data': processed_data,
            'classes': full_dataset.classes,
            'class_to_idx': full_dataset.class_to_idx
        }, output_path)

        print(f"\nОбработка завершена! Сохранено в {output_path}")
        return output_path

    def preprocess_dataset_MPPIC(self, dataset_path, output_path):
        """Сохраняет изображения с нарисованными скелетами руки"""
        raw_dataset = datasets.ImageFolder(dataset_path)

        progress_bar = tqdm(
            total=len(raw_dataset),
            desc="Обработка изображений",
            unit="img",
            dynamic_ncols=True
        )

        processed_data = []
        for i, (pil_image, label) in enumerate(raw_dataset):
            # Конвертируем PIL -> OpenCV (BGR)
            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            image_bgr = resize_to_640x480(image_bgr)
            # Пропускаем через MediaPipe
            skeleton_image = draw_hand_skeleton(image_bgr)

            # Преобразуем обратно в RGB
            image_rgb = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2RGB)
            pil_with_skeleton = transforms.ToPILImage()(image_rgb)
            tensor_image = self.transform(pil_with_skeleton)

            processed_data.append((tensor_image, label))

            progress_bar.update(1)
            progress_bar.set_postfix({
                "обработано": f"{i}/{len(raw_dataset)}",
                "текущий": f"{raw_dataset.classes[label]}"
            })

        progress_bar.close()
        random.shuffle(processed_data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            'data': processed_data,
            'classes': raw_dataset.classes,
            'class_to_idx': raw_dataset.class_to_idx
        }, output_path)

        print(f"\nОбработка завершена! Сохранено в {output_path}")
        return output_path


def draw_hand_skeleton(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Оборачиваем в mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Настройки модели
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        # Обработка изображения
        result = landmarker.detect(mp_image)

        # Рисуем скелет руки
        annotated_image = image_bgr.copy()
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks_on_image(annotated_image, hand_landmarks)
        else:
            print("Кисть не найдена.")

    return annotated_image


def draw_landmarks_on_image(image, hand_landmarks):
    # Используем стандартные связи между точками
    connections = solutions.hands.HAND_CONNECTIONS

    # Преобразуем в список координат
    h, w, _ = image.shape
    landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Рисуем точки
    for x, y in landmarks_px:
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    # Рисуем соединения
    for start_idx, end_idx in connections:
        x0, y0 = landmarks_px[start_idx]
        x1, y1 = landmarks_px[end_idx]
        cv2.line(image, (x0, y0), (x1, y1), (0, 255, 255), 2)

import mediapipe as mp

mp_hands = mp.solutions.hands

def extract_hand_features(image_bgr):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Преобразуем landmarks в вектор
            feature_vector = []
            for lm in hand_landmarks.landmark:
                feature_vector.extend([lm.x, lm.y, lm.z])
            return feature_vector
        else:
            print("⚠️ Рука не найдена")
            return None


def resize_to_640x480(image: np.ndarray) -> np.ndarray:
    resized_image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return resized_image
