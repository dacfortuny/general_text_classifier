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

# # News classifier

# This notebook is an example of the complete pipeline of the classifier model.

# Data used for this example is extracted from: https://www.kaggle.com/rmisra/news-category-dataset

# The code is based on the tutorial found in: https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/

# cd ..

# %load_ext autoreload
# %autoreload 2

# +
import pandas as pd

from sklearn.model_selection import train_test_split

from classifier.classifier_dataset import ClassifierDataset
from classifier.general_classifier import GeneralClassifier
# -

# # Settings

MODEL_NAME = "news"
RUN_NAME = "20210802_01"

DATA_FILE = "News_Category_Dataset_v2.json"

SEED_SHUFFLE = 31
VALIDATION_SIZE = 0.3
BERT_VERSION = "bert-base-cased"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = "cuda"
SEED_TRAIN_VALIDATION_SEPARATION = 31

SEED_TRAIN_TEST_SPLIT = 31
TEST_SIZE = 0.2

# ### Paths

DATA_FILE_PATH = f"data/{MODEL_NAME}/{DATA_FILE}"

MODEL_FILE_PATH = f"models/{MODEL_NAME}/{RUN_NAME}/model.pt"
MODEL_LOGS_PATH = f"models/{MODEL_NAME}/{RUN_NAME}"

# # Data

SAMPLE_SIZE = 1
def prepare_data_news_category(data_file):
    df = pd.read_json(data_file, lines=True)
    df["message"] = df["headline"] + " " + df["short_description"]
    df["label"] = df["category"].astype("category").cat.codes
    print(df[["label", "category"]].drop_duplicates().sort_values("label").reset_index(drop=True))
    df = df[["message", "label"]]
    df = df.sample(frac=SAMPLE_SIZE, random_state=SEED_SHUFFLE).reset_index(drop=True)
    df_train, df_test = train_test_split(df, random_state=SEED_TRAIN_TEST_SPLIT, test_size=TEST_SIZE, stratify=df["label"])
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


df_train, df_test = prepare_data_news_category(DATA_FILE_PATH)

# + tags=[]
dataset = ClassifierDataset(df_train, df_test, VALIDATION_SIZE, BATCH_SIZE, BERT_VERSION, SEED_TRAIN_VALIDATION_SEPARATION)
# -

# # Model

model = GeneralClassifier(dataset=dataset,
                          learning_rate=LEARNING_RATE,
                          epochs=EPOCHS,
                          best_model_file=MODEL_FILE_PATH,
                          device=DEVICE,
                          log_path=MODEL_LOGS_PATH)

# ## Training

# + tags=[]
model.train()

# + [markdown] tags=[]
# ## Performance report
# -

model.generate_performance_report()
