import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import torch
import torchvision.transforms as transforms
from models.model_definitions import SimpleNN, SimpleCNN, AdvancedCNN
import mediapipe as mp


class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Translator")
        self.root.geometry("800x600")

        # Инициализация модели
        self.model = None
        self.current_phrase = ""

        self.model_types = {
            "Многослойный персептрон": "simple_nn",
            "Простенькая сверточная": "simple_cnn",
            "Крутая сверточная": "advanced_cnn",
            "Простенькая CNN + MediaPipe": "simple_cnn_mp",
            "Крутая CNN + MediaPipe": "advanced_cnn_mp"
        }

        # Создание интерфейса
        self.create_menu()
        self.create_widgets()
        self.setup_camera()

        self.hand_landmarker = None
        self.mp_hands = mp.solutions.hands

        self.current_model_type = None

    def create_menu(self):
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Загрузить фото", command=self.load_image)
        file_menu.add_command(label="Сделать фото", command=self.capture_image)
        file_menu.add_separator()
        file_menu.add_command(label="Сбросить фразу", command=self.reset_phrase)
        menu_bar.add_cascade(label="Файл", menu=file_menu)

        model_menu = tk.Menu(menu_bar, tearoff=0)
        for model_name in self.model_types:
            model_menu.add_radiobutton(
                label=model_name,
                command=lambda name=model_name: self.select_model_type(name)
            )

        self.root.config(menu=menu_bar)
        menu_bar.add_cascade(label="Модель", menu=model_menu)

    def select_model_type(self, model_name):
        self.current_model_type = self.model_types[model_name]
        messagebox.showinfo("Выбор модели", f"Выбрана модель: {model_name}")
        self.load_model()

    def create_widgets(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Области для изображений
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.left_image_label = ttk.Label(self.image_frame)
        self.left_image_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.right_image_label = ttk.Label(self.image_frame)
        self.right_image_label.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Кнопка перевода
        self.translate_btn = ttk.Button(main_frame, text="Перевести", command=self.translate_image)
        self.translate_btn.pack(pady=10)

        # Текстовое поле
        self.phrase_text = tk.Text(main_frame, height=3, state=tk.DISABLED)
        self.phrase_text.pack(fill=tk.X, pady=10)

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showwarning("Предупреждение", "Камера не найдена!")
            self.camera_running = False
        else:
            self.camera_running = True

    def load_model(self):
        if not self.current_model_type:
            messagebox.showwarning("Предупреждение", "Сначала выберите тип модели!")
            return

        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            try:
                if "mp" in self.current_model_type:
                    self.init_mediapipe()

                if "simple_nn" in self.current_model_type:
                    self.model = SimpleNN(num_classes=29)
                elif "simple_cnn" in self.current_model_type:
                    self.model = SimpleCNN(num_classes=29)
                elif "advanced_cnn" in self.current_model_type:
                    self.model = AdvancedCNN(num_classes=29)

                # Загрузка весов
                self.model.load_state_dict(torch.load(file_path))
                self.model.eval()
                messagebox.showinfo("Успешно", "Модель загружена!")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def init_mediapipe(self):
        if self.hand_landmarker is None:
            self.hand_landmarker = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5)

    def preprocess_with_mediapipe(self, image):
        if self.hand_landmarker:
            results = self.hand_landmarker.process(image)
            # Добавьте здесь обработку ландмарок
            return image
        return image

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        if file_path:
            self.process_image(file_path)

    def capture_image(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite("captured_image.jpg", frame)
                self.process_image("captured_image.jpg")

    def process_image(self, image_path):
        try:
            image = Image.open(image_path)

            # Если выбрана модель с MediaPipe
            if "mp" in self.current_model_type:
                # Конвертация для MediaPipe
                mp_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                processed_image = self.preprocess_with_mediapipe(mp_image)
                image = Image.fromarray(processed_image)

            # Остальная обработка
            image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.left_image_label.config(image=photo)
            self.left_image_label.image = photo
            self.current_image = image_path

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки изображения: {str(e)}")

    def translate_image(self):
        if not self.model:
            messagebox.showerror("Ошибка", "Сначала выберите модель!")
            return

        if not self.current_image:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение!")
            return

        try:
            # Преобразование изображения для модели
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            image = Image.open(self.current_image).convert('RGB')
            tensor = transform(image).unsqueeze(0)

            # Предсказание
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_class = predicted.item()

            # Отображение результата
            self.show_prediction_dialog(predicted_class)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка распознавания: {str(e)}")

    def show_prediction_dialog(self, class_idx):
        class_mapping = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del',
            5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
            10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N',
            15: 'nothing', 16: 'O', 17: 'P', 18: 'Q',
            19: 'R', 20: 'S', 21: 'space', 22: 'T',
            23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'
        }
        predicted_char = class_mapping.get(class_idx, 'Неизвестно')

        dialog = tk.Toplevel(self.root)
        dialog.title("Результат распознавания")

        ttk.Label(dialog, text=f"Распознанный жест: {predicted_char}").pack(padx=20, pady=10)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Верно",
                   command=lambda: self.confirm_prediction(predicted_char, dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Неверно",
                   command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def confirm_prediction(self, char, dialog):
        self.current_phrase += char
        self.update_phrase_text()
        dialog.destroy()

    def update_phrase_text(self):
        self.phrase_text.config(state=tk.NORMAL)
        self.phrase_text.delete(1.0, tk.END)
        self.phrase_text.insert(tk.END, self.current_phrase)
        self.phrase_text.config(state=tk.DISABLED)

    def reset_phrase(self):
        self.current_phrase = ""
        self.update_phrase_text()

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = ASLTranslatorApp(root)
    root.mainloop()