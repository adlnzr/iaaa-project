# train.py

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.nn.functional import sigmoid
from collections import Counter
from tqdm import tqdm
from tester import Tester, Tester_AutoencoderClassification, Tester_ViT_smalldata
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast


'''
 class Trainer:
 
 data loader sends 4D data: [batch_size, 20 slices, h, w]
 each slice is passed to the model : [batch_size, 1, h, w] 
 loss, backward and updates are calculated for every single image,
 metrics are claculated based on max votting for 20 slices of each input (each patient)
 '''

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
        self.best_val_avgmetric = 0
        self.epochs_no_improve = 0
        self.save_path = save_path

        # Initialize the Tester instance
        self.tester = Tester(
            model=self.model,
            criterion=self.criterion,
            test_dl=self.val_dl,
            test_dataset=self.val_dataset,
            device=self.device,
            threshold=self.threshold
        )

        # initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=f"runs/training/{self.model.__class__.__name__}")

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_correct = 0.0

        for images, label in tqdm(self.train_dl):
            images = images.float().to(device=self.device)
            label = label.float().to(device=self.device)

            patient_outputs = []
            for i in range(images.size(1)):
                input = images[:, i, :, :]  
                input = input.unsqueeze(1)  

                # forward pass
                model_output = self.model(input)
                output = model_output.logits if hasattr(model_output, 'logits') else model_output
                loss = self.criterion(output, label.unsqueeze(1))

                patient_outputs.append(output)

                # backward and optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_preds = [sigmoid(x) for x in patient_outputs]
            train_stacked_preds = torch.stack(train_preds)
            train_mean_preds = torch.mean(train_stacked_preds, dim=0)
            train_result = (train_mean_preds > self.threshold).float()
            train_correct += (train_result == label.unsqueeze(1)).sum().item()

        train_accuracy = train_correct / len(self.train_dataset)

        return train_loss, train_accuracy

    def early_stopping(self, val_avgmetric):
        if val_avgmetric > self.best_val_avgmetric:
            self.best_val_avgmetric = val_avgmetric
            self.epochs_no_improve = 0
            torch.save(self.model.state_dict(), f"saved_models/{self.model.__class__.__name__}_best.pth")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print("Early stopping triggered")
            return True
        return False

    def train(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            train_loss, train_accuracy = self.train_one_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, val_avgmetric, conf_matrix = self.tester.test(phase="Val")

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Log the losses and metrics to TensorBoard
            self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
            }, epoch)

            self.writer.add_scalars('Accuracy', {
            'Train': train_accuracy,
            'Validation': val_accuracy
            }, epoch)
        
            self.writer.add_scalars('F1', {
            #    'Train': train_f1,
                'Validation': val_f1
            }, epoch)
            
            self.writer.add_scalars('Precision', {
            #    'Train': train_precision,
                'Validation': val_precision
            }, epoch)
            
            self.writer.add_scalars('Recall', {
            #    'Train': train_recall,
                'Validation': val_recall
            }, epoch)

            self.writer.add_scalars('avg_metric', {
            #    'Train': train_avgmetric,
                'Validation': val_avgmetric
            }, epoch)

            self.writer.add_scalars('auc', {
            #    'Train': train_auc,
                'Validation': val_auc
            }, epoch)
            

            if self.early_stopping(val_avgmetric):
                    break

        self.writer.close()
            
'''
Trainer_AutoencoderClassification

two loss functions in the model
so handling loss funcion has changed a bit 
'''

class Trainer_AutoencoderClassification:

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
        self.best_val_avgmetric = 0
        self.epochs_no_improve = 0
        self.save_path = save_path

        # Initialize the Tester instance
        self.tester = Tester_AutoencoderClassification(
            model=self.model,
            criterion=self.criterion,
            test_dl=self.val_dl,
            test_dataset=self.val_dataset,
            device=self.device,
            threshold=self.threshold
        )

        # initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=f"runs/training/{self.model.__class__.__name__}")

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_correct = 0.0

        for images, label in tqdm(self.train_dl):
            images = images.float().to(device=self.device)
            label = label.float().to(device=self.device)

            patient_outputs = []
            for i in range(images.size(1)):
                input = images[:, i, :, :]  
                input = input.unsqueeze(1)  

                # forward pass
                reconstructed, outputs = self.model(input) # two outputs for AutoencoderClassification model

                outputs = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # compute losses
                loss_reconstruction = self.criterion[0](reconstructed, input)
                loss_classification = self.criterion[1](outputs, label.unsqueeze(1))
                loss = loss_reconstruction + loss_classification

                patient_outputs.append(outputs)

                # backward and optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_preds = [sigmoid(x) for x in patient_outputs]
            train_stacked_preds = torch.stack(train_preds)
            train_mean_preds = torch.sum(train_stacked_preds, dim=0)
            train_result = (train_mean_preds > self.threshold).float()
            train_correct += (train_result == label.unsqueeze(1)).sum().item()

        train_accuracy = train_correct / len(self.train_dataset)

        return train_loss, train_accuracy

    def early_stopping(self, val_avgmetric):
        if val_avgmetric > self.best_val_avgmetric:
            self.best_val_avgmetric = val_avgmetric
            self.epochs_no_improve = 0
            torch.save(self.model.state_dict(), f"saved_models/{self.model.__class__.__name__}_best.pth")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print("Early stopping triggered")
            return True
        return False

    def train(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            train_loss, train_accuracy = self.train_one_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, val_avgmetric, conf_matrix = self.tester.test(phase="Val")
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Log the losses and metrics to TensorBoard

            self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
            }, epoch)

            self.writer.add_scalars('Accuracy', {
            'Train': train_accuracy,
            'Validation': val_accuracy
            }, epoch)
        
            self.writer.add_scalars('F1', {
            #    'Train': train_f1,
                'Validation': val_f1
            }, epoch)
            
            self.writer.add_scalars('Precision', {
            #    'Train': train_precision,
                'Validation': val_precision
            }, epoch)
            
            self.writer.add_scalars('Recall', {
            #    'Train': train_recall,
                'Validation': val_recall
            }, epoch)

            self.writer.add_scalars('avg_metric', {
            #    'Train': train_avgmetric,
                'Validation': val_avgmetric
            }, epoch)

            self.writer.add_scalars('auc', {
            #    'Train': train_auc,
                'Validation': val_auc
            }, epoch)
            

            if self.early_stopping(val_avgmetric):
                    break

        self.writer.close()


scaler = GradScaler('cuda')

class Trainer_ViT_smalldata:

    def __init__(self, model_1, model_2, criterion, optimizer, train_dl, val_dl, train_dataset, val_dataset, device, num_epochs, patience, threshold, save_path):
        self.model_1 = model_1
        self.model_2 = model_2
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
        self.best_val_avgmetric = 0
        self.epochs_no_improve = 0
        self.save_path = save_path

        # Initialize the Tester instance
        self.tester = Tester_ViT_smalldata(
            model_1 = self.model_1,
            model_2 = self.model_2,
            criterion=self.criterion,
            test_dl=self.val_dl,
            test_dataset=self.val_dataset,
            device=self.device,
            threshold=self.threshold
        )

        # initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=f"runs/training/{self.model_2.__class__.__name__}")

    def train_one_epoch(self):
        self.model_1.train()
        self.model_2.train()
        train_loss = 0.0
        train_correct = 0.0

        for images, label in tqdm(self.train_dl):
            images = images.float().to(device=self.device)
            label = label.float().to(device=self.device)
            with autocast(device_type='cuda'):

                patient_cls_outputs = []
                for i in range(images.size(1)):
                    input = images[:, i, :, :]  
                    input = input.unsqueeze(1)   # [8, 1, 224, 224]

                    # forward pass for each patient image
                    # output = model_output.logits if hasattr(model_output, 'logits') else model_output
                    cls_output = self.model_1(input)
                    patient_cls_outputs.append(cls_output)
                patient_cls_outputs = torch.stack(patient_cls_outputs, dim=1) # [bs, 20, 768]

                classifier_logit = self.model_2(patient_cls_outputs)
                loss = self.criterion(classifier_logit, label.unsqueeze(1))

            # backward and optimizer
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            train_loss += loss.item()

            train_result = (sigmoid(classifier_logit) > self.threshold).float()
            train_correct += (train_result == label.unsqueeze(1)).sum().item()

        train_accuracy = train_correct / len(self.train_dataset)

        return train_loss, train_accuracy

    def early_stopping(self, val_avgmetric):
        if val_avgmetric > self.best_val_avgmetric:
            self.best_val_avgmetric = val_avgmetric
            self.epochs_no_improve = 0
        
            # Save the parameters of both models separately
            torch.save(self.model_1.state_dict(), f'saved_models/{self.model_1.__class__.__name__}_best.pth')
            torch.save(self.model_2.state_dict(), f'saved_models/{self.model_2.__class__.__name__}_best.pth')

        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print("Early stopping triggered")
            return True
        return False

    def train(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            train_loss, train_accuracy = self.train_one_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc, val_avgmetric, conf_matrix = self.tester.test(phase="Val")

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Log the losses and metrics to TensorBoard
            self.writer.add_scalars('Loss', {
            'Train': train_loss,
            'Validation': val_loss
            }, epoch)

            self.writer.add_scalars('Accuracy', {
            'Train': train_accuracy,
            'Validation': val_accuracy
            }, epoch)
        
            self.writer.add_scalars('F1', {
            #    'Train': train_f1,
                'Validation': val_f1
            }, epoch)
            
            self.writer.add_scalars('Precision', {
            #    'Train': train_precision,
                'Validation': val_precision
            }, epoch)
            
            self.writer.add_scalars('Recall', {
            #    'Train': train_recall,
                'Validation': val_recall
            }, epoch)

            self.writer.add_scalars('avg_metric', {
            #    'Train': train_avgmetric,
                'Validation': val_avgmetric
            }, epoch)

            self.writer.add_scalars('auc', {
            #    'Train': train_auc,
                'Validation': val_auc
            }, epoch)
            

            if self.early_stopping(val_avgmetric):
                    break

        self.writer.close()
            

# class Trainer:

#     def __init__(self, model, criterion, optimizer, train_dl, val_dl, train_dataset, val_dataset, device, num_epochs, patience, threshold, save_path):
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.train_dl = train_dl
#         self.val_dl = val_dl
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.device = device
#         self.num_epochs = num_epochs
#         self.patience = patience
#         self.threshold = threshold
#         self.best_avg_metric = 0
#         self.epochs_no_improve = 0
#         self.save_path = save_path



#     def train_one_epoch(self):
#         self.model.train()
#         train_loss = 0.0
#         patient_correct = 0.0

#         for images, label in tqdm(self.train_dl):
#             images = images.float().to(device=self.device) # size: ([32,20,224,224])
#             label = label.float().to(device=self.device) # size: ([32])
    
#             patient_outputs = []
#             for i in range(images.size(1)):
#                 input = images[:, i, :, :] # size: ([32,224,224])
#                 input = input.unsqueeze(1) # size: ([32,1,224,224])

#                 # Forward pass
#                 output = self.model(input) # size: ([32,1])
#                 loss = self.criterion(output, label.unsqueeze(1)) # size: ([])
                
#                 patient_outputs.append(output) # keeping outputs of each patient in a list of tensors
#                 # len(patient_outputs) = 20 | list of tensors
                
#                 # Backward and optimizer
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 train_loss += loss.item()
            


#             patient_preds = [(sigmoid(x) > self.threshold).float() for x in patient_outputs] # list (20,32,1)
#             patient_max_vot = Counter(patient_preds).most_common(1)[0][0] # size: torch.tensor([32,1])
#             # (since using the Counter class, the result is a tensor, not a list.)

#             patient_correct += (patient_max_vot == label.unsqueeze(1)).sum().item()

#         train_accuracy = patient_correct / len(self.train_dataset)
        
#         return train_loss, train_accuracy


#     @torch.no_grad()
#     def evaluate(self):
#         self.model.eval()
#         val_correct = 0.0
#         all_labels = []
#         all_preds = []

#         for images, label in self.val_dl:
#             images = images.float().to(device=self.device)
#             label = label.float().to(device=self.device)

#             # Stack the list into a single tensor
#             # output_stacked = torch.stack(output_list, dim=0)
#             # outputs_sigmoid = torch.sigmoid(output_stacked)    
#             # outputs_predictions = (outputs_sigmoid > 0.5).float()
#             # final_prediction = torch.mode(outputs_predictions, dim=0)
#             # val_correct += (final_prediction == label).item()
#             # all_labels.extend(label.cpu().numpy())
#             # all_preds.extend(final_prediction.cpu().numpy())

#             output_list = []
#             for i in range(images.size(1)):
#                 input = images[:, i, :, :] # [32,224,224]
#                 input = input.unsqueeze(1) # [32,1,224,224]
#                 output = self.model(input)
#                 output_list.append(output)

#                 val_preds = [(sigmoid(x) > self.threshold).float() for x in output_list] # len= 20
#                 val_max_vot = Counter(val_preds).most_common(1)[0][0] # [32,1]      
#                 val_correct += (val_max_vot == label.unsqueeze(1)).sum().item()

#                 all_labels.extend(label.cpu().numpy())
#                 all_preds.extend(val_max_vot.squeeze(1).cpu().numpy())

#         val_accuracy = val_correct / len(self.val_dataset)
#         all_labels = np.array(all_labels)
#         all_preds = np.array(all_preds)


#         # check for empty predictions
#         if np.sum(all_preds) == 0:
#             precision = 0.0
#             recall = 0.0
#         else:
#             precision = precision_score(all_labels, all_preds)
#             recall = recall_score(all_labels, all_preds)

#         auc = roc_auc_score(all_labels, all_preds)
#         avg_metric = (precision + recall + val_accuracy) / 3

#         # Calculate and print the confusion matrix
#         conf_matrix = confusion_matrix(all_labels, all_preds)
#         print("Confusion Matrix:")
#         print(conf_matrix)

#         return val_accuracy, precision, recall, auc, avg_metric

#     def early_stopping(self, avg_metric):
#         if avg_metric > self.best_avg_metric:
#             self.best_avg_metric = avg_metric
#             self.epochs_no_improve = 0
#             # Save the best model
#             torch.save(self.model.state_dict(), self.save_path)
#         else:
#             self.epochs_no_improve += 1

#         if self.epochs_no_improve >= self.patience:
#             print("Early stopping triggered")
#             return True
#         return False

#     def train(self):
#         for epoch in range(self.num_epochs):
#             train_loss, train_accuracy = self.train_one_epoch()
#             val_accuracy, precision, recall, auc, avg_metric = self.evaluate()

#             print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
#             print(f"Epoch {epoch+1}/{self.num_epochs}, Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")

#             if self.early_stopping(avg_metric):
#                 break

# # train.py


# '''
# output = self.model(input).logits added
# '''

# class Trainer_for_resnet18:

#     def __init__(self, model, criterion, optimizer, train_dl, val_dl, train_dataset, val_dataset, device, num_epochs, patience, threshold, save_path):
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.train_dl = train_dl
#         self.val_dl = val_dl
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.device = device
#         self.num_epochs = num_epochs
#         self.patience = patience
#         self.threshold = threshold
#         self.best_avg_metric = 0
#         self.epochs_no_improve = 0
#         self.save_path = save_path



#     def train_one_epoch(self):
#         self.model.train()
#         train_loss = 0.0
#         patient_correct = 0.0

#         for images, label in tqdm(self.train_dl):
#             images = images.float().to(device=self.device) # size: ([32,20,224,224])
#             label = label.float().to(device=self.device) # size: ([32])
    
#             patient_outputs = []
#             for i in range(images.size(1)):
#                 input = images[:, i, :, :] # size: ([32,224,224])
#                 input = input.unsqueeze(1) # size: ([32,1,224,224])

#                 # forward pass
#                 output = self.model(input) # size: ([32,1])
#                 output = self.model(input).logits
#                 loss = self.criterion(output, label.unsqueeze(1)) # size: ([])
                
#                 patient_outputs.append(output) # keeping outputs of each patient in a list of tensors
#                 # len(patient_outputs) = 20 | list of tensors
                
#                 # backward and optimizer
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 train_loss += loss.item()
            
#             patient_preds = [(sigmoid(x) > self.threshold).float() for x in patient_outputs] # list (20,32,1)
#             patient_max_vot = Counter(patient_preds).most_common(1)[0][0] # size: torch.tensor([32,1])
#             # (since using the Counter class, the result is a tensor, not a list.)

#             patient_correct += (patient_max_vot == label.unsqueeze(1)).sum().item()

#         train_accuracy = patient_correct / len(self.train_dataset)
        
#         return train_loss, train_accuracy


#     @torch.no_grad()
#     def evaluate(self):
#         self.model.eval()
#         val_correct = 0.0
#         all_labels = []
#         all_preds = []

#         for images, label in self.val_dl:
#             images = images.float().to(device=self.device)
#             label = label.float().to(device=self.device)

#             output_list = []
#             for i in range(images.size(1)):
#                 input = images[:, i, :, :] # [32,224,224]
#                 input = input.unsqueeze(1) # [32,1,224,224]
#                 output = self.model(input)
#                 output = self.model(input).logits
#                 output_list.append(output)

#                 val_preds = [(sigmoid(x) > self.threshold).float() for x in output_list] # len= 20
#                 val_max_vot = Counter(val_preds).most_common(1)[0][0] # [32,1]      
#                 val_correct += (val_max_vot == label.unsqueeze(1)).sum().item()

#                 all_labels.extend(label.cpu().numpy())
#                 all_preds.extend(val_max_vot.squeeze(1).cpu().numpy())

#         val_accuracy = val_correct / (len(self.val_dataset) * 20)
#         all_labels = np.array(all_labels)
#         all_preds = np.array(all_preds)


#         # check for empty predictions
#         if np.sum(all_preds) == 0:
#             precision = 0.0
#             recall = 0.0
#         else:
#             precision = precision_score(all_labels, all_preds)
#             recall = recall_score(all_labels, all_preds)

#         auc = roc_auc_score(all_labels, all_preds)
#         avg_metric = (precision + recall + val_accuracy) / 3

#         # Calculate and print the confusion matrix
#         conf_matrix = confusion_matrix(all_labels, all_preds)
#         print("Confusion Matrix:")
#         print(conf_matrix)

#         return val_accuracy, precision, recall, auc, avg_metric

#     def early_stopping(self, avg_metric):
#         if avg_metric > self.best_avg_metric:
#             self.best_avg_metric = avg_metric
#             self.epochs_no_improve = 0
#             # Save the best model
#             torch.save(self.model.state_dict(), self.save_path)
#         else:
#             self.epochs_no_improve += 1

#         if self.epochs_no_improve >= self.patience:
#             print("Early stopping triggered")
#             return True
#         return False

#     def train(self):
#         for epoch in range(self.num_epochs):
#             train_loss, train_accuracy = self.train_one_epoch()
#             val_accuracy, precision, recall, auc, avg_metric = self.evaluate()

#             print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
#             print(f"Epoch {epoch+1}/{self.num_epochs}, Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")

#             if self.early_stopping(avg_metric):
#                 break



# class OldTrainer:
#     def __init__(self, model, criterion, optimizer, train_dl, val_dl, train_dataset, val_dataset, device, num_epochs, patience, threshold, save_path):
#         self.model = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.train_dl = train_dl
#         self.val_dl = val_dl
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.device = device
#         self.num_epochs = num_epochs
#         self.patience = patience
#         self.threshold = threshold
#         self.best_avg_metric = 0
#         self.epochs_no_improve = 0
#         self.save_path = save_path

#     def train_one_epoch(self):
#         self.model.train()
#         train_loss = 0.0
#         train_correct = 0.0

#         for images, labels in tqdm(self.train_dl):
#             images = images.float().to(device=self.device)
#             labels = labels.float().to(device=self.device)

#             # Forward pass
#             outputs = self.model(images)
#             loss = self.criterion(outputs, labels.unsqueeze(1))

#             # Backward and optimizer
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             train_loss += loss.item()
#             preds = (sigmoid(outputs) > self.threshold).float()
#             train_correct += (preds == labels).sum().item()

#         train_accuracy = train_correct / len(self.train_dataset)
#         return train_loss, train_accuracy

#     @torch.no_grad()
#     def evaluate(self):
#         self.model.eval()
#         val_correct = 0.0
#         all_labels = []
#         all_preds = []

#         for images, labels in self.val_dl:
#             images = images.float().to(device=self.device)
#             labels = labels.float().to(device=self.device)

#             outputs = self.model(images)
#             # Apply sigmoid to convert logits to probabilities
#             outputs = sigmoid(outputs)

#             preds = (outputs > self.threshold).float()
#             val_correct += (preds == labels).sum().item()

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(outputs.cpu().numpy())

#         val_accuracy = val_correct / len(self.val_dataset)
#         all_labels = np.array(all_labels)
#         all_preds = np.array(all_preds)
#         all_preds_binary = (all_preds > self.threshold).astype(float)

#         # Check for empty predictions
#         if np.sum(all_preds_binary) == 0:
#             precision = 0.0
#             recall = 0.0
#         else:
#             precision = precision_score(all_labels, all_preds_binary)
#             recall = recall_score(all_labels, all_preds_binary)

#         auc = roc_auc_score(all_labels, all_preds)
#         avg_metric = (precision + recall + auc) / 3

#         # Calculate and print the confusion matrix
#         conf_matrix = confusion_matrix(all_labels, all_preds_binary)
#         print("Confusion Matrix:")
#         print(conf_matrix)

#         return val_accuracy, precision, recall, auc, avg_metric

#     def early_stopping(self, avg_metric):
#         if avg_metric > self.best_avg_metric:
#             self.best_avg_metric = avg_metric
#             self.epochs_no_improve = 0
#             # Save the best model
#             torch.save(self.model.state_dict(), self.save_path)
#         else:
#             self.epochs_no_improve += 1

#         if self.epochs_no_improve >= self.patience:
#             print("Early stopping triggered")
#             return True
#         return False

#     def train(self):
#         for epoch in range(self.num_epochs):
#             train_loss, train_accuracy = self.train_one_epoch()
#             val_accuracy, precision, recall, auc, avg_metric = self.evaluate()

#             print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
#             print(f"Epoch {epoch+1}/{self.num_epochs}, Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")

#             if self.early_stopping(avg_metric):
#                 break
