import pandas as pd
import torch

from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class TensorDatapack(object):
    """
    Class containing the tensors of the ClassifierDataset.
    """

    def __init__(self, messages: pd.Series, labels: pd.Series, bert_version: str):
        """
        Class initializer.
        Args:
            messages: Series with all messages.
            labels: Series with all labels.
            bert_version: Bert version.
        """
        self.messages = messages
        self.labels = labels
        self.bert_version = bert_version

        self.tokenizer = self._define_tokenizer()
        tokens = self._tokenize_messages()

        self.token_ids = torch.tensor(tokens["input_ids"])
        self.attention_mask = torch.tensor(tokens["attention_mask"])
        self.labels = torch.tensor(labels.tolist())

    def _define_tokenizer(self) -> BertTokenizer:
        """
        Creates the tokenizer.
        Returns:
            Tokenizer for the given Bert version.
        """
        return BertTokenizer.from_pretrained(self.bert_version)

    def _tokenize_messages(self) -> BatchEncoding:
        """
        Tokenizes all messages.
        Returns:
            BatchEncoding object with tokenized messages.
        """
        return self.tokenizer.batch_encode_plus(self.messages.tolist(),
                                                padding="max_length",
                                                truncation=True,
                                                max_length=512,
                                                add_special_tokens=True,
                                                return_token_type_ids=False)
