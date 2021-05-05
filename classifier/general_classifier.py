import numpy as np
import torch
import torch.nn as nn

from classifier.architecture import ClassifierArchitecture
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import AutoModel, AdamW


class GeneralClassifier(object):

    def __init__(self, dataset, learning_rate, epochs, best_model_file, device, seed):

        self.dataset = dataset
        self.bert_version = dataset.bert_version
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.best_model_file = best_model_file
        self.model = self.define_architecture().to(device)
        self.optimizer = self.define_optimizer()
        self.loss_function = self.define_loss_function()

    def define_architecture(self):
        bert = AutoModel.from_pretrained(self.bert_version)
        for param in bert.parameters():
            param.requires_grad = False
        return ClassifierArchitecture(bert=bert, num_classes=len(self.dataset.classes))

    def define_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def define_loss_function(self):
        class_weights = compute_class_weight("balanced", np.unique(self.dataset.train_labels),
                                             self.dataset.train_labels)
        weights = torch.tensor(class_weights, dtype=torch.float)
        weights = weights.to(self.device)
        return nn.NLLLoss(weight=weights)

    def train_epoch(self):
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

    def evaluate_epoch(self):
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
        best_valid_loss = float("inf")
        for epoch in range(self.epochs):
            print(f"\n Epoch {epoch + 1} / {self.epochs}")
            train_loss = self.train_epoch()
            validation_loss = self.evaluate_epoch()
            if validation_loss < best_valid_loss:
                best_valid_loss = validation_loss
                torch.save(self.model.state_dict(), self.best_model_file)
            print(f"\nTraining Loss: {train_loss:.3f}")
            print(f"Validation Loss: {validation_loss:.3f}")

    def predict(self):
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
        self.model.load_state_dict(torch.load(self.best_model_file))
        predictions = self.predict()
        predictions_final = np.argmax(predictions, axis=1)
        test_labels = self.dataset.test_injector.tensor_datapack.labels
        print(classification_report(test_labels, predictions_final))
