import pandas as pd

from sklearn.model_selection import train_test_split

from classifier.tensor_datapack import TensorDatapack
from classifier.data_injector import DataInjector


class ClassifierDataset(object):
    """
    Class containing the data used to train the general classifier.
    """

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, validation_size: float, batch_size: int,
                 bert_version: str, seed: int):
        """
        Class initializer.
        Args:
            df_train: Dataframe with train data containing two columns: message (str) and label (int).
            df_test: Dataframe with test data containing two columns: message (str) and label (int).
            validation_size: Ratio of the training data used for validation.
            batch_size: Batch size of the training loop.
            bert_version: Bert version.
            seed: Seed for the train/validation split.
        """
        self.df_train = df_train
        self.df_test = df_test
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.bert_version = bert_version
        self.seed = seed

        self.classes = df_train["label"].unique().tolist()

        self.train_messages, self.validation_messages, self.train_labels, self.validation_labels = train_test_split(
            self.df_train["message"],
            self.df_train["label"],
            random_state=self.seed,
            test_size=self.validation_size,
            stratify=self.df_train["label"])
        self.test_messages = self.df_test["message"]
        self.test_labels = self.df_test["label"]

        self.train_injector = self._create_train_data_injector()
        self.validation_injector = self._create_validation_data_injector()
        self.test_injector = self._create_test_data_injector()

    def _create_train_data_injector(self) -> DataInjector:
        """
        Creates DataInjector with the training data.
        Returns:
            DataInjector for training data.
        """
        train_tensor_datapack = TensorDatapack(self.train_messages, self.train_labels, self.bert_version)
        return DataInjector(train_tensor_datapack, sampler_type="random", batch_size=self.batch_size)

    def _create_validation_data_injector(self) -> DataInjector:
        """
        Creates DataInjector with the validation data.
        Returns:
            DataInjector for validation data.
        """
        validation_tensor_datapack = TensorDatapack(self.validation_messages, self.validation_labels, self.bert_version)
        return DataInjector(validation_tensor_datapack, sampler_type="sequential", batch_size=self.batch_size)

    def _create_test_data_injector(self) -> DataInjector:
        """
        Creates DataInjector with the test data.
        Returns:
            DataInjector for test data.
        """
        test_tensor_datapack = TensorDatapack(self.test_messages, self.test_labels, self.bert_version)
        return DataInjector(test_tensor_datapack, sampler_type="sequential", batch_size=self.batch_size)
