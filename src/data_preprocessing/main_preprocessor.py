from data_preprocessing.mediapipe_preprocessor import ASLMPPreprocessor
from simple_preprocessor import ASLPreprocessor

RAW_DATA_PATH = r'D:\GitHubRepo\ASL-coursework\src\data\raw\ASL_Alphabet_Dataset\asl_alphabet_train'

preprocessor = ASLPreprocessor()
preprocessor.preprocess_dataset(RAW_DATA_PATH,
                                r'D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train.pt')

preprocessor = ASLMPPreprocessor()
preprocessor.preprocess_dataset_MPVEC(RAW_DATA_PATH,
                         r'D:\GitHubRepo\ASL-coursework\src\data\processed\ASL_Alphabet_Dataset\asl_alphabet_train_VEC.pt')
#preprocessor.preprocess_dataset_MPPIC(RAW_DATA_PATH,
#                                      PROCESSED_PATH)
