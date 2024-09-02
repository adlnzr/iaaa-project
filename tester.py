# test_file.py
'''
class Tester:
 
 data loader sends 4D data: [batch_size, 20 slices, h, w]
 each slice is passed to the model : [batch_size, 1, h, w] 

 metrics are claculated based on max votting for 20 slices of each input (each patient)

'''

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.nn.functional import sigmoid
from collections import Counter


class Tester:
    def __init__(self, model, criterion, test_dl, test_dataset, device, threshold=0.5):
        self.model = model
        self.criterion = criterion
        self.test_dl = test_dl
        self.test_dataset = test_dataset
        self.device = device
        self.threshold = threshold


    @torch.no_grad()
    def test(self, phase="Test"):
        self.model.eval()
        test_correct = 0.0
        test_loss = 0.0
        all_labels = []
        all_preds = []

        for images, label in self.test_dl:
            images = images.float().to(device=self.device)
            label = label.float().to(device=self.device)

            output_list = []
            for i in range(images.size(1)):
                input = images[:, i, :, :]
                input = input.unsqueeze(1)

                # forward pass
                model_output = self.model(input)
                output = model_output.logits if hasattr(model_output, 'logits') else model_output
                loss = self.criterion(output, label.unsqueeze(1))

                output_list.append(output)
                test_loss += loss.item()

            # Applying sigmoid and thresholding to each output in the list
            test_preds = [(sigmoid(x) > self.threshold).float() for x in output_list]  # len=20
            stacked_preds = torch.stack(test_preds)
            sum_preds = torch.sum(stacked_preds, dim=0)
            test_max_vot = (sum_preds > (len(test_preds) / 2)).float()  # tensor of shape [32, 1]
            test_correct += (test_max_vot == label.unsqueeze(1)).sum().item()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(test_max_vot.squeeze(1).cpu().numpy())

        test_accuracy = test_correct / len(self.test_dataset)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        if np.sum(all_preds) == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)

        auc = roc_auc_score(all_labels, all_preds)
        avg_metric = (precision + recall + test_accuracy) / 3
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f"{phase} Loss: {test_loss:.4f}, {phase} Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")

        # Print confusion matrix
        print("Confusion Matrix:")
        print(conf_matrix)

        if phase == "Val":
            return test_loss, test_accuracy, precision, recall, auc, avg_metric, conf_matrix


class Tester_AutoencoderClassification:
    def __init__(self, model, criterion, test_dl, test_dataset, device, threshold=0.5):
        self.model = model
        self.criterion = criterion
        self.test_dl = test_dl
        self.test_dataset = test_dataset
        self.device = device
        self.threshold = threshold


    @torch.no_grad()
    def test(self, phase="Test"):
        self.model.eval()
        test_correct = 0.0
        test_loss = 0.0
        all_labels = []
        all_preds = []

        for images, label in self.test_dl:
            images = images.float().to(device=self.device)
            label = label.float().to(device=self.device)

            output_list = []
            for i in range(images.size(1)):
                input = images[:, i, :, :]
                input = input.unsqueeze(1)

                # forward pass
                reconstructed, model_predict = self.model(input)
                
                model_predict = model_predict.logits if hasattr(model_predict, 'logits') else model_predict
                # compute losses
                loss_reconstruction = self.criterion[0](reconstructed, input)
                loss_classification = self.criterion[1](model_predict, label.unsqueeze(1))
                loss = loss_reconstruction + loss_classification

                output_list.append(model_predict)
                test_loss += loss.item()

            # as long as sigmoid has been applied to the model_predict in the model
            # no need any further sigmoid 
            test_preds = [(x > self.threshold).float() for x in output_list]  # len=20
            stacked_preds = torch.stack(test_preds)
            sum_preds = torch.sum(stacked_preds, dim=0)
            test_max_vot = (sum_preds > (len(test_preds) / 2)).float()  # tensor of shape [32, 1]
            test_correct += (test_max_vot == label.unsqueeze(1)).sum().item()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(test_max_vot.squeeze(1).cpu().numpy())

        test_accuracy = test_correct / len(self.test_dataset)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        if np.sum(all_preds) == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)

        auc = roc_auc_score(all_labels, all_preds)
        avg_metric = (precision + recall + test_accuracy) / 3
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f"{phase} Loss: {test_loss:.4f}, {phase} Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")

        # Print confusion matrix
        print("Confusion Matrix:")
        print(conf_matrix)

        if phase == "Val":
            return test_loss, test_accuracy, precision, recall, auc, avg_metric, conf_matrix

# class Tester:
#     def __init__(self, model, test_dl, test_dataset, device, threshold):
#         self.model = model
#         self.test_dl = test_dl
#         self.test_dataset = test_dataset
#         self.device = device
#         self.threshold = threshold

#     @torch.no_grad()
#     def test(self):
#         self.model.eval()
#         test_correct = 0.0
#         all_labels = []
#         all_preds = []

#         for images, label in self.test_dl:
#             images = images.float().to(device=self.device)
#             label = label.float().to(device=self.device)

#             output_list = []
#             for i in range(images.size(1)):
#                 input = images[:, i, :, :]
#                 input = input.unsqueeze(1)
#                 output = self.model(input)      
#                 output_list.append(output)

#                 test_preds = [(sigmoid(x) > self.threshold).float() for x in output_list] # len= 20
#                 test_max_vot = Counter(test_preds).most_common(1)[0][0] # [32,1]
#                 test_correct += (test_max_vot == label.unsqueeze(1)).sum().item()

#                 all_labels.extend(label.cpu().numpy())
#                 all_preds.extend(test_max_vot.squeeze(1).cpu().numpy())
            

#         test_accuracy = test_correct / len(self.test_dataset)
#         all_labels = np.array(all_labels)
#         all_preds = np.array(all_preds)

#         precision = precision_score(all_labels, all_preds)
#         recall = recall_score(all_labels, all_preds)
#         auc = roc_auc_score(all_labels, all_preds)

#         avg_metric = (precision + recall + test_accuracy) / 3

#         # print the metrics
#         print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")





# class OldTester:
#     def __init__(self, model, test_dl, test_dataset, device, threshold):
#         self.model = model
#         self.test_dl = test_dl
#         self.test_dataset = test_dataset
#         self.device = device
#         self.threshold = threshold

#     @torch.no_grad()
#     def test(self):
#         self.model.eval()
#         test_correct = 0.0
#         all_labels = []
#         all_preds = []

#         for images, labels in self.test_dl:
#             images = images.float().to(device=self.device)
#             labels = labels.float().to(device=self.device)

#             outputs = self.model(images)
#             outputs = sigmoid(outputs) # added by Adel
#             # convert logits to binary predictions
#             preds = (outputs > self.threshold).float()
#             test_correct += (preds == labels).sum().item()

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(outputs.cpu().numpy())

#         test_accuracy = test_correct / len(self.test_dataset)

#         all_labels = np.array(all_labels)
#         all_preds = np.array(all_preds)
#         all_preds_binary = (all_preds > self.threshold).astype(float)

#         precision = precision_score(all_labels, all_preds_binary)
#         recall = recall_score(all_labels, all_preds_binary)
#         auc = roc_auc_score(all_labels, all_preds)

#         avg_metric = (precision + recall + auc) / 3

#         # Print the metrics
#         print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")





# class OldTesterOneChannel:
#     def __init__(self, model, test_dl, test_dataset, device, threshold):
#         self.model = model
#         self.test_dl = test_dl
#         self.test_dataset = test_dataset
#         self.device = device
#         self.threshold = threshold

#     @torch.no_grad()
#     def test(self):
#         self.model.eval()
#         test_correct = 0.0
#         all_labels = []
#         all_preds = []

#         for images, labels in self.test_dl:
#             images = images.float().to(device=self.device)
#             labels = labels.float().to(device=self.device)

#             output_list = []
#             for i in range(images.size(1)):
#                 input = images[:, i, :, :]
#                 input = input.unsqueeze(1)

#                 output = self.model(input)      
#                 output_list.append(output)

#             # stack the list of tensors into a single tensor along a new dimension
#             stacked_output = torch.stack(output_list, dim=0) # [20, 32, 1]
            
#             final_output = torch.mean(stacked_output, dim=0) # [32, 1]
#             final_output = sigmoid(final_output)

#             preds = (final_output > self.threshold).float()
#             test_correct += (preds == labels).sum().item()

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(final_output.cpu().numpy())

#         test_accuracy = test_correct / len(self.test_dataset)
#         all_labels = np.array(all_labels)
#         all_preds = np.array(all_preds)

#         precision = precision_score(all_labels, all_preds)
#         recall = recall_score(all_labels, all_preds)
#         auc = roc_auc_score(all_labels, all_preds)

#         avg_metric = (precision + recall + test_accuracy) / 3

#         # Print the metrics
#         print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")