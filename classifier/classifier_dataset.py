from sklearn.model_selection import train_test_split

from classifier.tensor_datapack import TensorDatapack
from classifier.data_injector import DataInjector


class ClassifierDataset(object):

    def __init__(self, df_train, df_test, validation_size, batch_size, bert_version, seed):
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

        self.train_injector = self.create_train_data_injector()
        self.validation_injector = self.create_validation_data_injector()
        self.test_injector = self.create_test_data_injector()

    def create_train_data_injector(self):
        train_tensor_datapack = TensorDatapack(self.train_messages, self.train_labels, self.bert_version)
        return DataInjector(train_tensor_datapack, sampler_type="random", batch_size=self.batch_size)

    def create_validation_data_injector(self):
        validation_tensor_datapack = TensorDatapack(self.validation_messages, self.validation_labels, self.bert_version)
        return DataInjector(validation_tensor_datapack, sampler_type="sequential", batch_size=self.batch_size)

    def create_test_data_injector(self):
        test_tensor_datapack = TensorDatapack(self.test_messages, self.test_labels, self.bert_version)
        return DataInjector(test_tensor_datapack, sampler_type="sequential", batch_size=self.batch_size)
