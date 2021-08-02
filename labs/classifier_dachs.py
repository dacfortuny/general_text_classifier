# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Hate speech classifier

# This notebook is an example of the complete pipeline of the classifier model.

# Data used for this example is extracted from: https://zenodo.org/record/3520150

# The code is based on the tutorial found in: https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/

# cd ..

# %load_ext autoreload
# %autoreload 2

# +
import pandas as pd

from classifier.classifier_dataset import ClassifierDataset
from classifier.general_classifier import GeneralClassifier
# -

# ## Settings

MODEL_NAME = "dachs"
RUN_NAME = "20210802_07"

TRAIN_FILE = "Train_Hate_Messages.csv"
TEST_FILE = "Test_Hate_Messages.csv"

SEED_SHUFFLE = 31
VALIDATION_SIZE = 0.3
BERT_VERSION = "bert-base-multilingual-cased"
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 20
DEVICE = "cuda"
SEED_TRAIN_VALIDATION_SPLIT = 31

# ### Paths

TRAIN_FILE_PATH = f"data/{MODEL_NAME}/{TRAIN_FILE}"
TEST_FILE_PATH = f"data/{MODEL_NAME}/{TEST_FILE}"

MODEL_FILE_PATH = f"models/{MODEL_NAME}/{RUN_NAME}/model.pt"
MODEL_LOGS_PATH = f"models/{MODEL_NAME}/{RUN_NAME}"


# ## Data

def prepare_data_dachs(data_file: str) -> pd.DataFrame:
    """
    Read Dachs data from file and creates a datafram suitable for a ClassifierDataset object.
    Args:
        data_file: Path to file containing data.
    Returns:
        Dataframe with two columns: message and label
    """
    df = pd.read_csv(data_file, sep="|")
    df = df.dropna()
    df = df.rename(columns={"Hate_Speech": "label"})
    df = df[["message", "label"]]
    return df.sample(frac=1, random_state=SEED_SHUFFLE).reset_index(drop=True)


df_train = prepare_data_dachs(TRAIN_FILE_PATH)
df_test = prepare_data_dachs(TEST_FILE_PATH)

dataset = ClassifierDataset(df_train, df_test, VALIDATION_SIZE, BATCH_SIZE, BERT_VERSION, SEED_TRAIN_VALIDATION_SPLIT)

# ## Model

model = GeneralClassifier(dataset=dataset,
                          learning_rate=LEARNING_RATE,
                          epochs=EPOCHS,
                          best_model_file=MODEL_FILE_PATH,
                          device=DEVICE,
                          log_path=MODEL_LOGS_PATH)

# ### Training

model.train()

# ### Performance report

model.generate_performance_report()
