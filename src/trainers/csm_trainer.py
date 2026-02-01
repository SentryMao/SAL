import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import matplotlib.pyplot as plt

from src.evaluator.evaluator import EERMetric, F1Metric, EERMetricWithLabel, F1MetricWithLabel
from src.models.criterion.basic_loss import MaskedMultiClassCrossEntropyLoss
from src.trainers.label_generators import SPLLabelGenerator, TransitionLabelGenerator

class CSMTrainer(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            v2: int = 1,
            mixup: bool = False,
            mixup2: bool = False,
            mixup_ratio: float = 0.5,
            compile: bool = False,
            vis: bool = False,
            vis_top_k_hard: int = 10,
            vis_top_k_easy: int = 5,
    ) -> None:
        """Initialize the CSM trainer (mixing only).

        :param net: The model to train.
        :param criterion: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param mixup: Whether to apply mixup data augmentation.
        :param mixup_ratio: The ratio of samples in a batch to apply mixup to (0.0 to 1.0).
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        self.v2 = v2
        self.mixup = mixup
        self.mixup2 = mixup2
        self.mixup_ratio = mixup_ratio
        self.vis = vis
        self.vis_top_k_hard = vis_top_k_hard
        self.vis_top_k_easy = vis_top_k_easy
        # load model, criterion, optimizer, and scheduler
        self.net = net
        self.criterion = MaskedMultiClassCrossEntropyLoss(num_classes=8)
        self.criterion2 = MaskedMultiClassCrossEntropyLoss(num_classes=2)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking test metrics
        self.val_eer = EERMetric()
        self.val_eer_best = MinMetric()
        self.test_eer = EERMetric()
        self.val_acc = F1Metric()
        self.test_acc = F1Metric()

    def _mixup_batch(
            self,
            batch: Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply mixup data augmentation to a batch.
        
        Randomly selects a portion of the batch (based on mixup_ratio) and applies mixup
        by shuffling and concatenating from random cut points using probability distributions.
        
        :param batch: A batch of data containing (utt_ids, inputs, labels, label_lengths)
        :return: The batch with mixup applied to selected samples
        """
        utt_ids, inputs, labels, label_lengths = batch
        batch_size = inputs.size(0)
        
        
        # Calculate how many samples to apply mixup to
        num_mixup = int(batch_size * self.mixup_ratio)
        
        if num_mixup == 0:
            # No mixup to apply
            return batch
        
        # Create a random permutation of the batch indices
        perm_indices = torch.randperm(batch_size)
        
        # Select the first num_mixup samples for mixup
        mixup_indices = perm_indices[:num_mixup]
        original_indices = perm_indices[num_mixup:]
        
        # Create shuffled versions for mixup samples
        shuffled_inputs = inputs[mixup_indices]
        shuffled_labels = labels[mixup_indices]
        shuffled_label_lengths = label_lengths[mixup_indices]
        
        # Get sequence lengths for inputs and labels
        seq_len = inputs.size(1)
        max_label_len = labels.size(1)
        
        # Generate random cut points using uniform distribution
        # Ensure cut points are within valid range (avoiding 0 and full length)
        min_cut_ratio = 0.2  # Minimum 20% from start
        max_cut_ratio = 0.8  # Maximum 80% from start
        
        # Uniform distribution for cut points
        cut_ratios_input = torch.rand(num_mixup, device=inputs.device) * (max_cut_ratio - min_cut_ratio) + min_cut_ratio
        cut_ratios_label = torch.rand(num_mixup, device=inputs.device) * (max_cut_ratio - min_cut_ratio) + min_cut_ratio
        
        # Alternative: Normal distribution (uncomment to use instead of uniform)
        # mean_ratio = 0.5
        # std_ratio = 0.15
        # cut_ratios_input = torch.normal(mean_ratio, std_ratio, size=(num_mixup,), device=inputs.device)
        # cut_ratios_label = torch.normal(mean_ratio, std_ratio, size=(num_mixup,), device=inputs.device)
        # # Clamp to valid range
        # cut_ratios_input = torch.clamp(cut_ratios_input, min_cut_ratio, max_cut_ratio)
        # cut_ratios_label = torch.clamp(cut_ratios_label, min_cut_ratio, max_cut_ratio)
        
        # Calculate cut points for each sample in the batch
        cut_points_input = (cut_ratios_input * seq_len).long()
        cut_points_label = (cut_ratios_label * max_label_len).long()
        
        # Ensure cut points are at least 1 and at most seq_len-1
        cut_points_input = torch.clamp(cut_points_input, 1, seq_len - 1)
        cut_points_label = torch.clamp(cut_points_label, 1, max_label_len - 1)
        
        # Apply mixup to selected samples with individual cut points
        mixed_inputs_list = []
        mixed_labels_list = []
        
        for i in range(num_mixup):
            # Get cut points for this sample
            cut_point_input = cut_points_input[i]
            
            # Get actual label lengths for both original and shuffled samples
            orig_label_len = label_lengths[mixup_indices[i]]
            shuffled_label_len = shuffled_label_lengths[i]
            
            # Calculate label cut point based on actual label length
            cut_point_label = min(cut_points_label[i], orig_label_len, shuffled_label_len)
            
            # Ensure cut point is within valid range for labels
            cut_point_label = max(1, min(cut_point_label, min(orig_label_len, shuffled_label_len) - 1))
            
            # Mix inputs (use the same cut point as labels to maintain alignment)
            mixed_input = torch.cat([
                inputs[mixup_indices[i], :cut_point_input],
                shuffled_inputs[i, cut_point_input:]
            ], dim=0)
            mixed_inputs_list.append(mixed_input)
            
            # Mix labels based on actual label lengths
            mixed_label = torch.cat([
                labels[mixup_indices[i], :cut_point_label],
                shuffled_labels[i, cut_point_label:]
            ], dim=0)
            mixed_labels_list.append(mixed_label)
        
        # Pad sequences to maintain batch dimensions
        mixed_inputs = torch.stack(mixed_inputs_list)
        mixed_labels = torch.stack(mixed_labels_list)
        
        # Update label lengths for mixed samples
        mixed_label_lengths = torch.zeros_like(label_lengths[mixup_indices])
        
        for i in range(num_mixup):
            # Get actual label lengths for both original and shuffled samples
            orig_label_len = label_lengths[mixup_indices[i]]
            shuffled_label_len = shuffled_label_lengths[i]
            
            # Calculate the new label length after mixing
            # The mixed label length should be the sum of the two parts
            cut_point_label = min(cut_points_label[i], orig_label_len, shuffled_label_len)
            cut_point_label = max(1, min(cut_point_label, min(orig_label_len, shuffled_label_len) - 1))
            
            # First part: from original sample (up to cut point)
            first_part_len = cut_point_label
            # Second part: from shuffled sample (from cut point to its end)
            second_part_len = shuffled_label_len - cut_point_label
            
            # Total mixed label length
            mixed_label_lengths[i] = first_part_len + second_part_len
            
            # Ensure it doesn't exceed the maximum label length
            mixed_label_lengths[i] = min(mixed_label_lengths[i], max_label_len)
        
        # Combine mixed samples with original samples
        final_inputs = torch.zeros_like(inputs)
        final_labels = torch.zeros_like(labels)
        final_label_lengths = torch.zeros_like(label_lengths)
        
        # Place mixed samples back in their original positions
        final_inputs[mixup_indices] = mixed_inputs
        final_labels[mixup_indices] = mixed_labels
        final_label_lengths[mixup_indices] = mixed_label_lengths
        
        # Keep original samples for the rest
        if len(original_indices) > 0:
            final_inputs[original_indices] = inputs[original_indices]
            final_labels[original_indices] = labels[original_indices]
            final_label_lengths[original_indices] = label_lengths[original_indices]
        
        return utt_ids, final_inputs, final_labels, final_label_lengths

    def _get_label_mask(
            self,
            labels: torch.Tensor,
            label_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Get a mask for the labels based on their lengths.

        :param labels: A tensor of labels.
        :param label_lengths: A tensor of label lengths.
        :return: A mask tensor where 1 indicates valid positions and 0 indicates
            padded positions.
        """
        mask = torch.ones_like(labels, dtype=torch.bool)
        for i, length in enumerate(label_lengths):
            mask[i, length:] = False
        return mask

    def _get_pred_label(
            self,
            preds: torch.Tensor,
            labels: torch.Tensor,
            label_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the predicted labels and their corresponding ground truth labels.

        :param preds: A tensor of model predictions.
        :param labels: A tensor of ground truth labels.
        :param label_lengths: A tensor of label lengths.
        :return: A tuple containing a list of predicted labels and a list of
            ground truth labels.
        """
        pred_list = []
        label_list = []
        for i, length in enumerate(label_lengths):
            pred_list.append(preds[i, :length])
            label_list.append(labels[i, :length])

        preds_flat = torch.cat(pred_list, dim=0)
        labels_flat = torch.cat(label_list, dim=0).float()
        return preds_flat, labels_flat

    @property
    def exp_dir(self) -> str:
        if hasattr(self.trainer, "ckpt_path") and self.trainer.ckpt_path:
            # Use the directory of the checkpoint file as the root
            return str(Path(self.trainer.ckpt_path).parent.parent)
        else:
            # Fallback to logger's log_dir
            log_dir = self.logger.log_dir
            return str.join("/", log_dir.split("/")[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_eer.reset()
        self.val_eer_best.reset()
        self.val_acc.reset()

    def training_step(
            self, batch: Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the
        training set.

        :param batch: A batch of data (a tuple) containing the input tensor
        of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Apply mixup if enabled
        if self.mixup:
            batch = self._mixup_batch(batch)
        if self.mixup2:
            batch = self._mixup_batch(batch)
            batch = self._mixup_batch(batch)
        utt_ids, inputs, labels, label_lengths = batch
        
        preds1, preds2 = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        
        # Generate boundary labels for batch
        labels_info, lengths_info = SPLLabelGenerator._seg2bd_label_new(labels)
        target_batch = SPLLabelGenerator.seg2bd_label_new(labels_info, lengths_info)
        
        # Convert batch targets to tensor format
        if isinstance(target_batch, list):
            # Handle batch data
            target_list = []
            for i, target_seq in enumerate(target_batch):
                target_tensor = torch.tensor(target_seq, device=preds1.device).squeeze()
                target_list.append(target_tensor)
            target = torch.cat(target_list, dim=0).type(torch.long)
        else:
            # Handle single sequence
            target = torch.tensor(target_batch, device=preds1.device).view(-1).type(torch.long)
        
        loss1 = self.criterion(preds1.transpose(1, 2),
                             target.reshape(labels.shape[0], -1).to(torch.long), mask=mask)
        loss2 = self.criterion2(preds2,
                             labels.to(torch.long), mask=mask)
        loss = loss2
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=True,
                 sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.train_loss.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, list],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the
        validation set.

        :param batch: A batch of data (a tuple) containing the input tensor
        of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        utt_ids, inputs, labels, label_lengths = batch
        
        preds1, preds2 = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        
        # Generate boundary labels for batch
        labels_info, lengths_info = SPLLabelGenerator._seg2bd_label_new(labels)
        target_batch = SPLLabelGenerator.seg2bd_label_new(labels_info, lengths_info)
        
        # Convert batch targets to tensor format
        if isinstance(target_batch, list):
            # Handle batch data
            target_list = []
            for i, target_seq in enumerate(target_batch):
                target_tensor = torch.tensor(target_seq, device=preds1.device).squeeze()
                target_list.append(target_tensor)
            target = torch.cat(target_list, dim=0).type(torch.long)
        else:
            # Handle single sequence
            target = torch.tensor(target_batch, device=preds1.device).view(-1).type(torch.long)
        loss1 = self.criterion(preds1.transpose(1, 2),
                              target.reshape(labels.shape[0], -1).to(torch.long), mask=mask)
        loss2 = self.criterion2(preds2,
                              labels.to(torch.long), mask=mask)
        loss = loss2
        # update and log metrics
        preds_flat, labels_flat = self._get_pred_label(preds2, labels,
                                                       label_lengths)
        self.val_loss(loss)
        self.val_eer.update(preds_flat[:, 1] - preds_flat[:, 0], labels_flat)
        self.val_acc.update(preds_flat, labels_flat)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        eer, thresh = self.val_eer.compute()  # get current val eer
        self.val_eer_best(eer)  # update best so far val eer
        acc, f1 = self.val_acc.compute()
        self.log("val/eer", eer, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val/eer_best", self.val_eer_best.compute(),
                 prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        self.val_loss.reset()
        self.val_eer.reset()
        self.val_acc.reset()

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        self.test_loss.reset()
        self.test_eer.reset()
        self.test_acc.reset()
        if self.vis:
            self._vis_items = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, list],
                  batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor
        of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        utt_ids, inputs, labels, label_lengths = batch
        preds1, preds2 = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        
        # Generate boundary labels for batch
        labels_info, lengths_info = SPLLabelGenerator._seg2bd_label_new(labels)
        target_batch = SPLLabelGenerator.seg2bd_label_new(labels_info, lengths_info)
        
        # Convert batch targets to tensor format
        if isinstance(target_batch, list):
            # Handle batch data
            target_list = []
            for i, target_seq in enumerate(target_batch):
                target_tensor = torch.tensor(target_seq, device=preds1.device).squeeze()
                target_list.append(target_tensor)
            target = torch.cat(target_list, dim=0).type(torch.long)
        else:
            # Handle single sequence
            target = torch.tensor(target_batch, device=preds1.device).view(-1).type(torch.long)
        
        loss1 = self.criterion(preds1.transpose(1, 2),
                              target.reshape(labels.shape[0], -1).to(torch.long), mask=mask)
        loss2 = self.criterion2(preds2,
                              labels.to(torch.long), mask=mask)
        loss = loss2
        # update and log metrics
        preds_flat, labels_flat = self._get_pred_label(preds2, labels,
                                                       label_lengths)
        preds_flat8, labels_flat8 = self._get_pred_label(preds1, target.reshape(labels.shape[0], -1),
                                                       label_lengths)
        self.test_loss(loss)
        
        self.test_eer.update(preds_flat[:, 1] - preds_flat[:, 0], labels_flat)
        self.test_acc.update(preds_flat, labels_flat)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        if self.vis:
            with torch.no_grad():
                B = labels.shape[0]
                for i in range(B):
                    L = int(label_lengths[i].item())
                    if L <= 0:
                        continue
                    scores = (preds2[i, :L, 1] - preds2[i, :L, 0]).detach().cpu()
                    labs = labels[i, :L].to(torch.long).detach().cpu()
                    utt_id = utt_ids[i] if isinstance(utt_ids, (list, tuple)) else str(i)
                    self._vis_items.append({"utt_id": utt_id, "scores": scores, "labels": labs})

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        eer, thresh_eer = self.test_eer.compute()
        acc, f1 = self.test_acc.compute()
        self.log("test/eer", eer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        result_file = os.path.join(self.exp_dir, "result.txt")
        with open(result_file, "a") as f:
            f.write(f"EER={eer:.4f} ACC={acc:.4f} F1={f1:.4f}\n")

        if self.vis and hasattr(self, "_vis_items") and len(self._vis_items) > 0:
            vis_dir = os.path.join(self.exp_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            # compute per-utt error rate
            errors = []
            for item in self._vis_items:
                scores = item["scores"].numpy()
                labs = item["labels"].numpy().astype(int)
                preds = (scores > float(thresh_eer)).astype(int)
                err = float(np.mean(np.array((preds != labs)))) if labs.size > 0 else 1.0
                errors.append(err)
            ranked = list(enumerate(errors))
            ranked.sort(key=lambda x: x[1], reverse=True)
            hard_idx = [i for i, _ in ranked[:self.vis_top_k_hard]]
            easy_idx = [i for i, _ in sorted(ranked[-self.vis_top_k_easy:], key=lambda x: x[1])]
            for tag, idx_list in (("hard", hard_idx), ("easy", easy_idx)):
                for pos, idx in enumerate(idx_list, start=1):
                    item = self._vis_items[idx]
                    scores = item["scores"].numpy()
                    labs = item["labels"].numpy().astype(int)
                    T = labs.shape[0]
                    xs = np.arange(T)
                    plt.figure(figsize=(12, 4))
                    plt.plot(xs, scores, color='tab:blue', linewidth=1.5, label='score (pos-neg)')
                    plt.axhline(float(thresh_eer), color='gray', linestyle='--', label=f'thresh={float(thresh_eer):.3f}')
                    ones = np.where(labs == 1)[0]
                    zeros = np.where(labs == 0)[0]
                    ax = plt.gca()
                    y_min, y_max = ax.get_ylim()
                    if ones.size > 0:
                        plt.scatter(ones, np.full_like(ones, y_max, dtype=float), s=8, c='tab:green', label='GT=1')
                    if zeros.size > 0:
                        plt.scatter(zeros, np.full_like(zeros, y_min, dtype=float), s=8, c='tab:red', label='GT=0')
                    plt.title(f"{tag.upper()} | utt={item['utt_id']} | err={errors[idx]:.3f}")
                    plt.xlabel("frame")
                    plt.ylabel("score / GT")
                    plt.legend(loc='best')
                    plt.tight_layout()
                    save_path = os.path.join(vis_dir, f"{tag}_{pos:02d}_{str(item['utt_id']).replace('/', '_')}.png")
                    plt.savefig(save_path, dpi=200)
                    plt.close()

        self.test_loss.reset()
        self.test_eer.reset()
        self.test_acc.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
                                                                                                                      
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizers and learning-rate schedulers to be used

        :return: A dict containing the configured optimizers and
        learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.net.parameters())

        if self.hparams.scheduler is not None:
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
