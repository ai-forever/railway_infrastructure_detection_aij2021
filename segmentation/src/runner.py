import torch
import torch.nn.functional as F
from catalyst import dl


class CustomRunner(dl.SupervisedRunner):
    def handle_batch(self, batch):
        x, y = batch['image'], batch['mask']
        logits = self.model(x)

        softm_preds = logits.softmax(dim=1)
        one_hot_targets = F.one_hot(
            y.squeeze(1).to(torch.int64), num_classes=4).permute(0, 3, 1, 2)

        self.batch = {
            "scores": logits,
            'mask': y,
            "softm_preds": softm_preds,
            "one_hot_targets": one_hot_targets,

        }