import random
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
        dataset = datasets.ImageFolder(dataset_path, transform=self.transform)

        # Создаем прогресс-бар
        progress_bar = tqdm(
            total=len(dataset),
            desc="Обработка изображений",
            unit="img",
            dynamic_ncols=True
        )

        processed_data = []
        for i, (image, label) in enumerate(dataset):
            processed_data.append((image, label))

            if i % 10 == 0:
                progress_bar.update(10)
                progress_bar.set_postfix({
                    "обработано": f"{i}/{len(dataset)}",
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