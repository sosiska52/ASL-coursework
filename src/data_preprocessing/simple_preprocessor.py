import random
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms, datasets
import os
from tqdm import tqdm


class ASLPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_dataset(self, dataset_path, output_path):
        """Сохраняет предобработанные данные с индикатором прогресса"""
        dataset = datasets.ImageFolder(dataset_path)

        # Создаем прогресс-бар
        progress_bar = tqdm(
            total=len(dataset),
            desc="Обработка изображений",
            unit="img",
            dynamic_ncols=True
        )

        processed_data = []
        for i, (image_pil, label) in enumerate(dataset):
            # Преобразуем PIL → OpenCV (BGR), ресайзим, потом обратно
            image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            image_bgr = resize_to_640x480(image_bgr)

            # Назад в RGB → PIL
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            # Трансформация → тензор
            image_tensor = self.transform(image_pil)
            processed_data.append((image_tensor, label))

            # Обновляем прогресс
            progress_bar.update(1)
            progress_bar.set_postfix({
                "обработано": f"{i+1}/{len(dataset)}",
                "текущий": f"{dataset.classes[label]}"
            })

        progress_bar.close()

        random.shuffle(processed_data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            'data': processed_data,
            'classes': dataset.classes,
            'class_to_idx': dataset.class_to_idx
        }, output_path)

        print(f"\nОбработка завершена! Сохранено в {output_path}")
        return output_path


def resize_to_640x480(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
