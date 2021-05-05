import torch

from transformers import BertTokenizer


class TensorDatapack(object):

    def __init__(self, messages, labels, bert_version):
        self.messages = messages
        self.labels = labels
        self.bert_version = bert_version

        self.tokenizer = self.define_tockenizer()
        tokens = self.tokenize_messages()

        self.token_ids = torch.tensor(tokens["input_ids"])
        self.attention_mask = torch.tensor(tokens["attention_mask"])
        self.labels = torch.tensor(labels.tolist())

    def define_tockenizer(self):
        return BertTokenizer.from_pretrained(self.bert_version)

    def tokenize_messages(self):
        return self.tokenizer.batch_encode_plus(self.messages.tolist(),
                                                padding="max_length",
                                                add_special_tokens=True,
                                                return_token_type_ids=False)
