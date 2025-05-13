from preprocess import ASLPreprocessor

RAW_DATA_PATH = r'D:\GitHubRepo\ASL-coursework\src\data\raw\ASL_Alphabet_Dataset\asl_alphabet_train'
PROCESSED_PATH = r'D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train'

preprocessor = ASLPreprocessor()
preprocessor.preprocess_dataset(RAW_DATA_PATH,PROCESSED_PATH)