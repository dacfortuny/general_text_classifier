# General text classifier

This repo contains code to train a text classifier.

## Steps to train the model

First, get the data and prepare them in two pandas dataframes (train and test) with two columns: **messages** (str) and **label** (int).

Secondly, create a **ClassifierDataset** running the following line:

`dataset = ClassifierDataset(df_train, df_test, VALIDATION_SIZE, BATCH_SIZE, BERT_VERSION, SEED_TRAIN_VALIDATION_SPLIT)`

where `VALIDATION_SIZE` is the ratio of the train set used for validation, `BATCH_SIZE` is the batch size for the training, `BERT_VERSION` is the bert version used and `SEED_TRAIN_VALIDATION_SPLIT` is the seed used for the train/validation random split.

Then,  Instantiate the model using:

`model = GeneralClassifier(dataset = dataset, learning_rate = LEARNING_RATE, epochs = EPOCHS, best_model_file = BEST_MODEL_FILE, device = DEVICE)`

where `LEARNING RATE` is the learning rate for the training, `EPOCHS` the number of epochs to train, `best_model_file` the path to the trained model file and `DEVICE` the device to use for the training ('cpu' or 'cuda').

Finally, train the model running:

`model.train()`

To obtain a report of the performance of the model based on the data of the test set, run the following instruction:

`model.generate_performance_report()`

## Example

An example of the usage of this library is found in [labs/classifier_dachs.py](https://github.com/dacfortuny/general_bert_classifier/blob/main/labs/classifier_dachs.py).

## Run in Google Colab

### Clone repository

```
from google.colab import drive
drive.mount("/content/gdrive")
```
```
cd gdrive/MyDrive
```
```
!git clone https://github.com/dacfortuny/general_text_classifier.git
```

### Configure notebook

Add these lines at the beginning of the notebook

```
from google.colab import drive
drive.mount("/content/gdrive")
```
```
cd gdrive/MyDrive/general_text_classifier
```
```
!pip install transformers
```
