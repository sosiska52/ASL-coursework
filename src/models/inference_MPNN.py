import cv2
import torch
import numpy as np
from torchvision.datasets.folder import default_loader

from models.model_definitions import MPNN
from data_preprocessing.mediapipe_preprocessor import extract_hand_features, resize_to_640x480

MODEL_PATH = r"D:\GitHubRepo\ASL-coursework\src\models\mpnn.pth"
PROCESSED_DATA_PATH = r"D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train_VEC.pt"

def predict_single(feature_vector):
    # === Загрузка модели и метаданных ===
    data = torch.load(PROCESSED_DATA_PATH)
    class_names = data['classes']
    input_dim = len(feature_vector)

    model = MPNN(input_dim=input_dim, num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # === Прогноз ===
    with torch.no_grad():
        input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    print(f"Модель предполагает: {predicted_class}")
    return predicted_class

# Пример использования:
image_pil = default_loader(r"D:\GitHubRepo\ASL-coursework\src\data\raw\ASL_Alphabet_Dataset\asl_alphabet_test\O_test.jpg")  # загружаем изображение вручную
image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
image_bgr = resize_to_640x480(image_bgr)
feature_vec = extract_hand_features(image_bgr)
cv2.imwrite("debug_image.jpg", image_bgr)
if feature_vec is not None:
    predict_single(feature_vec)
else:
    print("❌ Не удалось извлечь признаки с изображения!")

