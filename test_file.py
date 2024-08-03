# test_file.py

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score


class Tester:
    def __init__(self, model, test_dl, test_dataset, device, threshold):
        self.model = model
        self.test_dl = test_dl
        self.test_dataset = test_dataset
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def test(self):
        self.model.eval()
        test_correct = 0.0
        all_labels = []
        all_preds = []

        for images, labels in self.test_dl:
            images = images.float().to(device=self.device)
            labels = labels.float().to(device=self.device)

            outputs = self.model(images)
            # convert logits to binary predictions
            preds = (outputs > self.threshold).float()
            test_correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

        test_accuracy = test_correct / len(self.test_dataset)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_preds_binary = (all_preds > self.threshold).astype(float)

        precision = precision_score(all_labels, all_preds_binary)
        recall = recall_score(all_labels, all_preds_binary)
        auc = roc_auc_score(all_labels, all_preds)

        avg_metric = (precision + recall + auc) / 3

        # Print the metrics
        print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {precision:.4f}, Recall: {
              recall:.4f}, AUC: {auc:.4f}, Avg Metric: {avg_metric:.4f}")
