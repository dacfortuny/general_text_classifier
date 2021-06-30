from torch import Tensor

import torch.nn as nn


class ClassifierArchitecture(nn.Module):
    """
    Class with the architecture of the neural network.
    """

    def __init__(self, bert: str, num_classes: int):
        """
        Class initializer.
        Args:
            bert: Bert version.
            num_classes: Number of classes of the classifier.
        """
        super(ClassifierArchitecture, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass of the neural network.
        Args:
            sent_id: Tensor with the ids of the words.
            mask: Tensor with the mask that indicates which tokens are padding.
        Returns:
            Tensor with trained weights.
        """
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
