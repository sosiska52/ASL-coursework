import torch
from preprocess import ASLPreprocessor

RAW_DATA_PATH = "D:/path/to/asl_alphabet_train"
PROCESSED_PATH = "D:/path/to/processed/asl_processed.pt"

preprocessor = ASLPreprocessor()
preprocessor.preprocess_dataset(
    r'D:\GitHubRepo\ASL-coursework\src\data\raw\ASL_Alphabet_Dataset\asl_alphabet_train',
    r'D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train')

processed_data = torch.load(PROCESSED_PATH)