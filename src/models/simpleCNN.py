import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

DATA_PATH = r'D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train'
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
NUM_CLASSES = len(dataset.classes)
print(f"Найдено классов: {NUM_CLASSES}")  # Выведет что-то вроде "Найдено классов: 29"


class SimpleASLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 128),  # 32x32 пикселя, 3 канала RGB
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)  # Выходной слой по числу классов
        )

    def forward(self, x):
        return self.layers(x)


# 3. Обучение (упрощённое)
model = SimpleASLNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Простой SGD

# 4. Простейший тренировочный цикл
for epoch in range(5):  # Всего 5 эпох
    for images, labels in DataLoader(dataset, batch_size=32):
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Эпоха {epoch + 1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), r"D:\GitHubRepo\ASL-coursework\src\models")