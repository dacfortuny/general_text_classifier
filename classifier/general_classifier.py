import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AdamW

from classifier.classifier_architecture import ClassifierArchitecture
from classifier.classifier_dataset import ClassifierDataset


class GeneralClassifier(object):
    """
    Class with the general classifier model.
    """

    def __init__(self, dataset: ClassifierDataset, learning_rate: float, epochs: int, best_model_file: str,
                 device: str, log_path: str):
        """
        Class initializer.
        Args:
            dataset: ClassifierDataset object with train, validation and test data.
            learning_rate: Learning rate for the training.
            epochs: Number of epochs to train.
            best_model_file: Output path of the trained model.
            device: Device in which perform the calculations.
            log_path: Path to store the TensorBoard logs.
        """

        self.dataset = dataset
        self.bert_version = dataset.bert_version
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.best_model_file = best_model_file
        self.model = self.define_architecture().to(device)
        self.optimizer = self.define_optimizer()
        self.loss_function = self.define_loss_function()
        self.writer = SummaryWriter(log_path)

    def define_architecture(self) -> ClassifierArchitecture:
        """
        Creates the final architecture of the classifier model.
        Returns:
            ClassifierArchitecture object with the architecture of the neural network.
        """
        bert = AutoModel.from_pretrained(self.bert_version)
        for param in bert.parameters():
            param.requires_grad = False
        return ClassifierArchitecture(bert=bert, num_classes=len(self.dataset.classes))

    def define_optimizer(self) -> AdamW:
        """
        Creates the optimizer for the training.
        Returns:
            Optimizer object.
        """
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def define_loss_function(self) -> nn.NLLLoss:
        """
        Creates the loss function for the training.
        Returns:
            Loss function.
        """
        class_weights = compute_class_weight("balanced", np.unique(self.dataset.train_labels),
                                             self.dataset.train_labels)
        weights = torch.tensor(class_weights, dtype=torch.float)
        weights = weights.to(self.device)
        return nn.NLLLoss(weight=weights)

    def _train_epoch(self, epoch: int) -> float:
        """
        Performs the training for one epoch. Updates the weights and returns the current loss.
        Args:
            epoch: Number of the training epoch.
        Returns:
            Average loss after training the epoch.
        """
        print("\nTraining...")
        self.model.train()
        train_dataloader = self.dataset.train_injector.dataloader
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = [r.to(self.device) for r in batch]
            ids, mask, labels = batch
            self.model.zero_grad()
            predictions = self.model(ids, mask)
            loss = self.loss_function(predictions, labels)
            total_loss = total_loss + loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        average_loss = total_loss / len(train_dataloader)
        return average_loss

    def _evaluate_epoch(self) -> float:
        """
        Performs the validation for one epoch. Updates the weights and returns the current loss.
        Returns:
            Average loss of the validation set after training the epoch.
        """
        print("\nEvaluating...")
        self.model.eval()
        validation_dataloader = self.dataset.validation_injector.dataloader
        total_loss, total_accuracy = 0, 0
        for step, batch in enumerate(tqdm(validation_dataloader)):
            batch = [t.to(self.device) for t in batch]
            ids, mask, labels = batch
            with torch.no_grad():
                predictions = self.model(ids, mask)
                loss = self.loss_function(predictions, labels)
                total_loss = total_loss + loss.item()
        average_loss = total_loss / len(validation_dataloader)
        return average_loss

    def train(self):
        """
        Performs the complete training of the model.
        """
        best_valid_loss = float("inf")
        for epoch in range(self.epochs):
            print(f"\n Epoch {epoch + 1} / {self.epochs}")
            train_loss = self._train_epoch(epoch)
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            validation_loss = self._evaluate_epoch()
            self.writer.add_scalar("Loss/validation", validation_loss, epoch)
            if validation_loss < best_valid_loss:
                best_valid_loss = validation_loss
                torch.save(self.model.state_dict(), self.best_model_file)
            print(f"\nTraining Loss: {train_loss:.3f}")
            print(f"Validation Loss: {validation_loss:.3f}")

    def _predict(self) -> np.ndarray:
        """
        Perform predictions for the test dataset.
        Returns:
            Numpy array with predictions.
        """
        self.model.eval()
        test_dataloader = self.dataset.test_injector.dataloader
        total_predictions = []
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch = [t.to(self.device) for t in batch]
            ids, mask, labels = batch
            with torch.no_grad():
                predictions = self.model(ids, mask)
                predictions = predictions.detach().cpu().numpy()
                total_predictions.append(predictions)
        total_predictions = np.concatenate(total_predictions, axis=0)
        return total_predictions

    def generate_performance_report(self):
        """
        Prints a report based on the predictions of the test set.
        """
        self.model.load_state_dict(torch.load(self.best_model_file))
        predictions = self._predict()
        predictions_final = np.argmax(predictions, axis=1)
        test_labels = self.dataset.test_injector.tensor_datapack.labels
        report = classification_report(test_labels, predictions_final)
        self.writer.add_text("Classification report", report)
        print(report)
