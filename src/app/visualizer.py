import matplotlib.pyplot as plt
import seaborn as sns
import os


class TrainingVisualizer:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 14

    def update(self, train_loss, val_loss, train_acc=None, val_acc=None):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        if train_acc is not None:
            self.train_acc.append(train_acc)
        if val_acc is not None:
            self.val_acc.append(val_acc)

    def plot(self, save_path=None):
        epochs = range(1, len(self.train_loss) + 1)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss, 'b', label='Training loss')
        plt.plot(epochs, self.val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if self.train_acc and self.val_acc:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.train_acc, 'b', label='Training acc')
            plt.plot(epochs, self.val_acc, 'r', label='Validation acc')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def reset(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []