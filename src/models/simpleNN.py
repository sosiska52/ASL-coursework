import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from visualizer.visualizer import TrainingVisualizer
from models.model_definitions import SimpleNN

EPOCH = 20

class PreprocessedDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict['data']
        self.classes = data_dict['classes']
        self.class_to_idx = data_dict['class_to_idx']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


print("Loading data...")
PROCESSED_PATH = r'D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train'
loaded_data = torch.load(PROCESSED_PATH)
full_dataset = PreprocessedDataset(loaded_data)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

visualizer = TrainingVisualizer()
print("DONE")
start_time = time.time()

if __name__ == "__main__":
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        visualizer.update(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )

        print(f'Epoch [{epoch + 1}/{EPOCH}]')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n')

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"Общее время обучения: {minutes} мин {seconds} сек")

    visualizer.plot(save_path=r'D:\GitHubRepo\ASL-coursework\reports\figures\training_plot.png')
    torch.save(model.state_dict(), r"D:\GitHubRepo\ASL-coursework\src\models\simpleNN.pth")