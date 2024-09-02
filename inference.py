import numpy as np
import torch
from torch.nn.functional import sigmoid


def inference(model, data_loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in data_loader:
            images = images.float().to(device=device)

            output_list = []

            for i in range(images.size(1)):
                input = images[:, i, :, :]
                input = input.unsqueeze(1)

                model_output = model(input)
                output = model_output.logits if hasattr(
                    model_output, 'logits') else model_output

                output_list.append(output)

            test_preds = [sigmoid(x) for x in output_list]
            stacked_preds = torch.stack(test_preds)
            test_preds = torch.mean(stacked_preds, dim=0)

            all_preds.extend(test_preds.squeeze(1).cpu().numpy())

    return all_preds
