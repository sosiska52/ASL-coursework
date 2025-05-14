import torch
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from models.model_definitions import SimpleCNN
from visualizer.visualizer import TrainingVisualizer

# === Параметры ===
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
PROCESSED_PATH = r'D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train'
SAVE_MODEL_PATH = r"D:\GitHubRepo\ASL-coursework\src\models\simpleCNN.pth"
SAVE_PLOT_PATH = r'D:\GitHubRepo\ASL-coursework\reports\figures\training_plot_cnn.png'

# === Кастомный Dataset ===
class PreprocessedDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = data_dict['data']
        self.classes = data_dict['classes']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# === Аугментации и нормализация ===
transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # для 3 каналов
])

# === Загрузка данных ===
print("Загрузка данных...")
loaded_data = torch.load(PROCESSED_PATH)
dataset = PreprocessedDataset(loaded_data, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

visualizer = TrainingVisualizer()
print("Готово.")
start_time = time.time()

# === Обучение ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    num_classes = len(loaded_data['classes'])
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        # === Валидация ===
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        # === Метрики ===
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        visualizer.update(train_loss=avg_train_loss, val_loss=avg_val_loss,
                          train_acc=train_acc, val_acc=val_acc)

        print(f"[{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # === Финал ===
    total_time = time.time() - start_time
    print(f"Время обучения: {int(total_time // 60)} мин {int(total_time % 60)} сек")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    visualizer.plot(save_path=SAVE_PLOT_PATH)
