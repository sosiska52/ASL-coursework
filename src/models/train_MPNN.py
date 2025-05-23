import torch
import time
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from visualizer.visualizer import TrainingVisualizer
from models.model_definitions import MPNN

# === Параметры ===
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PROCESSED_DATA_PATH = r"D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train_VEC.pt"
MODEL_SAVE_PATH = r"D:\GitHubRepo\ASL-coursework\src\models\mpnn.pth"
PLOT_SAVE_PATH = r"D:\GitHubRepo\ASL-coursework\reports\figures\training_plot_mpnn.png"

# === Кастомный Dataset ===
class FeatureVectorDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict['data']
        self.classes = data_dict['classes']
        self.class_to_idx = data_dict['class_to_idx']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vec, label = self.data[idx]
        return torch.tensor(vec, dtype=torch.float32), label

# === Загрузка данных ===
print("Загрузка данных...")
loaded_data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
full_dataset = FeatureVectorDataset(loaded_data)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

visualizer = TrainingVisualizer()
print("Готово.")
start_time = time.time()

# === Обучение ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    input_dim = len(full_dataset[0][0])  # Размерность вектора признаков
    num_classes = len(loaded_data['classes'])

    model = MPNN(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for vectors, labels in tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS}", leave=False):
            vectors, labels = vectors.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(vectors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # === Валидация ===
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for vectors, labels in val_loader:
                vectors, labels = vectors.to(device), labels.to(device)
                outputs = model(vectors)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        visualizer.update(train_loss=avg_train_loss, val_loss=avg_val_loss,
                          train_acc=train_acc, val_acc=val_acc)

        print(f"[{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # === Сохранение ===
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    visualizer.plot(save_path=PLOT_SAVE_PATH)
    print("Обучение завершено.")
