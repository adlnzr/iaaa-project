# train.py
'''
Trainer -> for 18-channel image dataset

TrainerOneChannel -> for 1-channel image dataset

'''


import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.nn.functional import sigmoid
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, train_dl, val_dl, train_dataset, val_dataset, device, num_epochs, patience, threshold, save_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.threshold = threshold
        self.best_avg_metric = 0
        self.epochs_no_improve = 0
        self.save_path = save_path

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_correct = 0.0

        for images, labels in tqdm(self.train_dl):
            images = images.float().to(device=self.device)
            labels = labels.float().to(device=self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels.unsqueeze(1))

            # Backward and optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            preds = (sigmoid(outputs) > self.threshold).float()
            train_correct += (preds == labels).sum().item()

        train_accuracy = train_correct / len(self.train_dataset)
        return train_loss, train_accuracy

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        val_correct = 0.0
        all_labels = []
        all_preds = []

        for images, labels in self.val_dl:
            images = images.float().to(device=self.device)
            labels = labels.float().to(device=self.device)

            outputs = self.model(images)
            # Apply sigmoid to convert logits to probabilities
            outputs = sigmoid(outputs)

            preds = (outputs > self.threshold).float()
            val_correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

        val_accuracy = val_correct / len(self.val_dataset)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_preds_binary = (all_preds > self.threshold).astype(float)

        # Check for empty predictions
        if np.sum(all_preds_binary) == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = precision_score(all_labels, all_preds_binary)
            recall = recall_score(all_labels, all_preds_binary)

        auc = roc_auc_score(all_labels, all_preds)
        avg_metric = (precision + recall + auc) / 3

        # Calculate and print the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds_binary)
        print("Confusion Matrix:")
        print(conf_matrix)

        return val_accuracy, precision, recall, auc, avg_metric

    def early_stopping(self, avg_metric):
        if avg_metric > self.best_avg_metric:
            self.best_avg_metric = avg_metric
            self.epochs_no_improve = 0
            # Save the best model
            torch.save(self.model.state_dict(), self.save_path)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print("Early stopping triggered")
            return True
        return False

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_accuracy, precision, recall, auc, avg_metric = self.evaluate()

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch+1}/{self.num_epochs}, Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")

            if self.early_stopping(avg_metric):
                break


'''
 one channel for each image, therefore mean of 20 outputs of each patient is 
 considered for the loss function  
 '''

class TrainerOneChannel:

    def __init__(self, model, criterion, optimizer, train_dl, val_dl, train_dataset, val_dataset, device, num_epochs, patience, threshold, save_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.threshold = threshold
        self.best_avg_metric = 0
        self.epochs_no_improve = 0
        self.save_path = save_path



    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_correct = 0.0

        for images, labels in tqdm(self.train_dl):
            images = images.float().to(device=self.device)
            labels = labels.float().to(device=self.device)

            output_list = []
            for i in range(images.size(1)):
                input = images[:, i, :, :]
                input = input.unsqueeze(1)

                # Forward pass
                output = self.model(input)      
                output_list.append(output)

            # stack the list of tensors into a single tensor along a new dimension
            stacked_output = torch.stack(output_list, dim=0) # [20, 32, 1]
            
            final_output = torch.mean(stacked_output, dim=0) # [32, 1]

            loss = self.criterion(final_output, labels.unsqueeze(1))
            
            # Backward and optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            preds = (sigmoid(final_output) > self.threshold).float()
            train_correct += (preds == labels).sum().item()

        train_accuracy = train_correct / len(self.train_dataset)
        return train_loss, train_accuracy

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        val_correct = 0.0
        all_labels = []
        all_preds = []

        for images, labels in self.val_dl:
            images = images.float().to(device=self.device)
            labels = labels.float().to(device=self.device)

            output_list = []
            for i in range(images.size(1)):
                input = images[:, i, :, :] # [32,224,224]
                input = input.unsqueeze(1) # [32,1,224,224]
                output = self.model(input)

                output_list.append(output)

            stacked_output = torch.stack(output_list, dim=0)

            final_output = torch.mean(stacked_output, dim=0)

            
            # apply sigmoid to convert logits to probabilities
            final_output = sigmoid(final_output)

            preds = (final_output > self.threshold).float()
            val_correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(final_output.cpu().numpy())

        val_accuracy = val_correct / len(self.val_dataset)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_preds_binary = (all_preds > self.threshold).astype(float)

        # Check for empty predictions
        if np.sum(all_preds_binary) == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = precision_score(all_labels, all_preds_binary)
            recall = recall_score(all_labels, all_preds_binary)

        auc = roc_auc_score(all_labels, all_preds)
        avg_metric = (precision + recall + auc) / 3

        # Calculate and print the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds_binary)
        print("Confusion Matrix:")
        print(conf_matrix)

        return val_accuracy, precision, recall, auc, avg_metric

    def early_stopping(self, avg_metric):
        if avg_metric > self.best_avg_metric:
            self.best_avg_metric = avg_metric
            self.epochs_no_improve = 0
            # Save the best model
            torch.save(self.model.state_dict(), self.save_path)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print("Early stopping triggered")
            return True
        return False

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_accuracy, precision, recall, auc, avg_metric = self.evaluate()

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch+1}/{self.num_epochs}, Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")

            if self.early_stopping(avg_metric):
                break
