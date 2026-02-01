import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import matplotlib.pyplot as plt

from src.evaluator.evaluator import EERMetric, F1Metric, F1MetricWithLabel
from src.models.criterion.basic_loss import MaskedMultiClassCrossEntropyLoss
from src.trainers.label_generators import SPLLabelGenerator


class BaselineTrainer(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            compile: bool = False,
            vis: bool = False,
            vis_top_k_hard: int = 10,
            vis_top_k_easy: int = 5,
    ) -> None:
        """Initialize a trainer.

        :param net: The model to train.
        :param criterion: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        # load model, criterion, optimizer, and scheduler
        self.net = net
        self.criterion = MaskedMultiClassCrossEntropyLoss(num_classes=2)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vis = vis
        self.vis_top_k_hard = vis_top_k_hard
        self.vis_top_k_easy = vis_top_k_easy

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
        self.test_f1_with_label = F1MetricWithLabel()

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
        utt_ids, inputs, labels, label_lengths = batch
        preds = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        loss = self.criterion(preds.transpose(1, 2),
                             labels.to(torch.long), mask=mask)

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
        preds = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        loss = self.criterion(preds.transpose(1, 2),
                             labels.to(torch.long), mask=mask)

        # update and log metrics
        preds_flat, labels_flat = self._get_pred_label(preds, labels,
                                                       label_lengths)
        self.val_loss(loss)
        self.val_eer.update(preds_flat[:, 1], labels_flat)
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
        if self.vis:
            self._vis_items = []
        # Initialize list to store utterance error rates for sorting
        self._utt_error_rates = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, list],
                  batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor
        of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        utt_ids, inputs, labels, label_lengths = batch
        preds = self.forward(inputs)
        # Generate boundary labels for batch
        labels_info, lengths_info = SPLLabelGenerator._seg2bd_label_new(labels)
        target_batch = SPLLabelGenerator.seg2bd_label_new(labels_info, lengths_info)
        
        # Convert batch targets to tensor format
        if isinstance(target_batch, list):
            # Handle batch data
            target_list = []
            for i, target_seq in enumerate(target_batch):
                target_tensor = torch.tensor(target_seq, device=preds.device).squeeze()
                target_list.append(target_tensor)
            target = torch.cat(target_list, dim=0).type(torch.long)
        else:
            # Handle single sequence
            target = torch.tensor(target_batch, device=preds.device).view(-1).type(torch.long)
        mask = self._get_label_mask(labels, label_lengths)
        loss = self.criterion(preds.transpose(1, 2),
                             labels.to(torch.long), mask=mask)

        # update and log metrics
        preds_flat, labels_flat = self._get_pred_label(preds, labels,
                                                       label_lengths)
        self.test_loss(loss)
        self.test_eer.update(preds_flat[:, 1], labels_flat)
        self.test_acc.update(preds_flat, labels_flat)
        self.test_f1_with_label.update(preds_flat, labels_flat, target)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Collect utterance error rates for sorting (always collect, not just for visualization)
        with torch.no_grad():
            B = labels.shape[0]
            for i in range(B):
                L = int(label_lengths[i].item())
                if L <= 0:
                    continue
                seq_scores = preds[i, :L, 1].detach().cpu()
                seq_labels = labels[i, :L].to(torch.long).detach().cpu()
                utt_id = utt_ids[i] if isinstance(utt_ids, (list, tuple)) else str(i)
                
                # Calculate error rate for this utterance
                scores_np = seq_scores.numpy()
                labs_np = seq_labels.numpy().astype(int)
                # Use a simple threshold of 0.5 for error rate calculation
                preds_bin = (scores_np > 0.5).astype(int)
                err_rate = float(np.mean(np.array((preds_bin != labs_np)))) if labs_np.size > 0 else 1.0
                
                # Store utterance error rate information
                self._utt_error_rates.append({
                    "utt_id": utt_id, 
                    "error_rate": err_rate,
                    "scores": seq_scores, 
                    "labels": seq_labels
                })
                
                # Also store for visualization if enabled
                if self.vis:
                    self._vis_items.append({"utt_id": utt_id, "scores": seq_scores, "labels": seq_labels})

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        eer, thresh_eer = self.test_eer.compute()
        acc, f1 = self.test_acc.compute()
        acc2, f12, acc_with_label = self.test_f1_with_label.compute()
        self.log("test/eer", eer, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        # save the result
        result_file = os.path.join(self.exp_dir, "result.txt")
        with open(result_file, "a") as f:
            f.write(f"EER={eer:.4f} ACC={acc:.4f} F1={f1:.4f} Thresh={thresh_eer:.4f}\n")
            # 写入 acc_with_label 双重dict
            for main_key, subdict in acc_with_label.items():
                f.write(f"{main_key}:")
                for sub_key, value in subdict.items():
                    f.write(f" {sub_key}: {value:.4f},")
                f.write("\n")

        # visualization of hardest/easiest utterances
        if self.vis and hasattr(self, "_vis_items") and len(self._vis_items) > 0:
            vis_dir = os.path.join(self.exp_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Use the already computed error rates from _utt_error_rates
            if hasattr(self, "_utt_error_rates") and len(self._utt_error_rates) > 0:
                # Create a mapping from utt_id to error rate for visualization items
                error_rate_map = {item["utt_id"]: item["error_rate"] for item in self._utt_error_rates}
                
                # Get error rates for visualization items
                errors = []
                for item in self._vis_items:
                    utt_id = item["utt_id"]
                    err = error_rate_map.get(utt_id, 1.0)  # Default to 1.0 if not found
                    errors.append(err)
                
                ranked = list(enumerate(errors))
                ranked.sort(key=lambda x: x[1], reverse=True)
                hard_idx = [i for i, _ in ranked[:self.vis_top_k_hard]]
                easy_k = max(0, min(self.vis_top_k_easy, len(ranked)))
                easy_idx = [i for i, _ in sorted(ranked[-easy_k:], key=lambda x: x[1])]
                
                for tag, idx_list in (("hard", hard_idx), ("easy", easy_idx)):
                    for pos, idx in enumerate(idx_list, start=1):
                        item = self._vis_items[idx]
                        scores = item["scores"].numpy()
                        labs = item["labels"].numpy().astype(int)
                        T = labs.shape[0]
                        xs = np.arange(T)
                        plt.figure(figsize=(12, 4))
                        # draw score and EER threshold
                        plt.plot(xs, scores, color='tab:blue', linewidth=1.5, label='score (pos)')
                        plt.axhline(float(thresh_eer), color='gray', linestyle='--', label=f'thresh={float(thresh_eer):.3f}')
                        # GT points at top/bottom of axis
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
                        out_path = os.path.join(vis_dir, f"{tag}_{pos:02d}_{str(item['utt_id']).replace('/', '_')}.png")
                        plt.savefig(out_path, dpi=200)
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

