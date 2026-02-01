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

class SPLTrainer(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            v2: int = 1,
            compile: bool = False,
            vis: bool = False,
            vis_top_k_hard: int = 10,
            vis_top_k_easy: int = 5,
    ) -> None:
        """Initialize the SPL trainer (2-loss with SPL labels).

        :param net: The model to train.
        :param criterion: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        self.v2 = v2
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
        
        self.val_eer_with_label = EERMetricWithLabel()
        self.test_eer_with_label = EERMetricWithLabel()
        self.val_f1_with_label = F1MetricWithLabel()
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
        preds1, preds2 = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        
        # Generate SPL boundary labels for batch
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
        loss = loss1 + self.v2 * loss2
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
        
        # Generate SPL boundary labels for batch
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
        loss = loss1 + self.v2 * loss2
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
        # prepare containers for visualization if enabled
        if self.vis:
            self._vis_items = []  # list of dicts: {"utt_id", "scores", "labels"}
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
        preds1, preds2 = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        
        # Generate SPL boundary labels for batch
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
        loss = loss1 + self.v2 * loss2
        # update and log metrics
        preds_flat, labels_flat = self._get_pred_label(preds2, labels,
                                                       label_lengths)
        preds_flat8, labels_flat8 = self._get_pred_label(preds1, target.reshape(labels.shape[0], -1),
                                                       label_lengths)
        self.test_loss(loss)
        
        self.test_eer.update(preds_flat[:, 1] - preds_flat[:, 0], labels_flat)
        self.test_acc.update(preds_flat, labels_flat)
        self.test_eer_with_label.update(preds_flat[:, 1] - preds_flat[:, 0], labels_flat, labels_flat8)
        self.test_f1_with_label.update(preds_flat, labels_flat, labels_flat8)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Collect utterance error rates for sorting (always collect, not just for visualization)
        with torch.no_grad():
            B = labels.shape[0]
            for i in range(B):
                L = int(label_lengths[i].item())
                if L <= 0:
                    continue
                seq_scores = (preds2[i, :L, 1] - preds2[i, :L, 0]).detach().cpu()
                seq_labels = labels[i, :L].to(torch.long).detach().cpu()
                utt_id = utt_ids[i] if isinstance(utt_ids, (list, tuple)) else str(i)
                
                # Calculate error rate for this utterance
                scores_np = seq_scores.numpy()
                labs_np = seq_labels.numpy().astype(int)
                # Use a simple threshold of 0.0 for error rate calculation (since scores are already pos-neg)
                preds_bin = (scores_np > 0.0).astype(int)
                err_rate = float(np.mean(np.array((preds_bin != labs_np)))) if labs_np.size > 0 else 1.0
                
                # Store utterance error rate information
                self._utt_error_rates.append({
                    "utt_id": utt_id, 
                    "error_rate": err_rate,
                    "scores": seq_scores, 
                    "labels": seq_labels
                })

        # collect per-utterance scores and labels for visualization
        if self.vis:
            with torch.no_grad():
                batch_size = labels.shape[0]
                for i in range(batch_size):
                    length = int(label_lengths[i].item())
                    if length <= 0:
                        continue
                    seq_scores = (preds2[i, :length, 1] - preds2[i, :length, 0]).detach().cpu()
                    seq_labels = labels[i, :length].to(torch.long).detach().cpu()
                    utt_id = utt_ids[i] if isinstance(utt_ids, (list, tuple)) else str(i)
                    self._vis_items.append({
                        "utt_id": utt_id,
                        "scores": seq_scores,
                        "labels": seq_labels,
                        "input": inputs[i].detach().cpu(),
                        "length": length,
                    })

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        print("Start computing eer")
        eer, thresh_eer = self.test_eer.compute()
        print("EER: ", eer)
        acc, f1 = self.test_acc.compute()
        print("ACC: ", acc)
        print("F1: ", f1)
        
        acc2, f12, acc_with_label = self.test_f1_with_label.compute()
        print("ACC2: ", acc2)
        print("F12: ", f12)
        print("ACC_WITH_LABEL: ", acc_with_label)
        self.log("test/eer", eer, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        
        # save the result
        result_file = os.path.join(self.exp_dir, "result.txt")
        print(f"EER={eer:.4f} ACC={acc:.4f} F1={f1:.4f} Thresh={thresh_eer:.4f}\n")
        with open(result_file, "a") as f:
            f.write(f"EER={eer:.4f} ACC={acc:.4f} F1={f1:.4f} Thresh={thresh_eer:.4f}\n")
            # 写入 acc_with_label 双重dict
            for main_key, subdict in acc_with_label.items():
                f.write(f"{main_key}:")
                for sub_key, value in subdict.items():
                    f.write(f" {sub_key}: {value:.4f},")
                f.write("\n")

        # visualization of hardest/easiest utterances (Grad-CAM on seq model)
        if self.vis and hasattr(self, "_vis_items") and len(self._vis_items) > 0:
            vis_dir = os.path.join(self.exp_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            # compute per-utt error rate at threshold
            diffs = []
            for item in self._vis_items:
                scores = item["scores"].numpy()
                labels_np = item["labels"].numpy().astype(int)
                preds_bin = (scores > float(thresh_eer)).astype(int)
                err = float(np.mean(np.array((preds_bin != labels_np)))) if labels_np.size > 0 else 1.0
                diffs.append(err)
            # select indices
            indexed = list(enumerate(diffs))
            indexed.sort(key=lambda x: x[1], reverse=True)
            hard_indices = [idx for idx, _ in indexed[:self.vis_top_k_hard]]
            easy_k = max(0, min(self.vis_top_k_easy, len(indexed)))
            easy_indices = [idx for idx, _ in sorted(indexed[-easy_k:], key=lambda x: x[1])]
            select_pairs = [("hard", hard_indices), ("easy", easy_indices)]

            # Prepare helper for Grad-CAM
            class _SeqIOHook:
                def __init__(self):
                    self.input_act = None
                    self.output_act = None
                    self.input_grad = None
                    self.output_grad = None
                def fwd(self, module, inp, out):
                    self.input_act = inp[0].detach()
                    self.output_act = (out[0] if isinstance(out, (tuple, list)) else out).detach()
                def bwd(self, module, grad_in, grad_out):
                    self.input_grad = grad_in[0].detach() if grad_in is not None and len(grad_in) > 0 else None
                    gout = grad_out[0] if grad_out is not None and len(grad_out) > 0 else None
                    self.output_grad = ((gout[0] if isinstance(gout, (tuple, list)) else gout).detach() if gout is not None else None)

            def _temporal_gradcam(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
                # [B,T,D]
                weights = grads.mean(dim=2, keepdim=True)
                cam = (weights * acts).sum(dim=2)
                cam = F.relu(cam)
                cam_min = cam.amin(dim=1, keepdim=True)
                cam_max = cam.amax(dim=1, keepdim=True)
                return (cam - cam_min) / (cam_max - cam_min + 1e-7)

            for tag, indices in select_pairs:
                for rank, idx in enumerate(indices, start=1):
                    item = self._vis_items[idx]
                    utt_id = item['utt_id']
                    x = item['input'].unsqueeze(0).to(self.device).float()
                    y_np = item['labels'].numpy().astype(float)

                    # Forward pass and register hooks on seq model
                    self.net.eval()
                    assert hasattr(self.net, 'seq_model'), 'Model must have attribute seq_model for Grad-CAM.'
                    hook = _SeqIOHook()
                    fh = self.net.seq_model.register_forward_hook(hook.fwd)
                    bh = self.net.seq_model.register_full_backward_hook(hook.bwd)
                    try:
                        self.net.zero_grad(set_to_none=True)
                        out1, out2 = self.net(x)
                        # choose target time and class from 8-class head
                        probs = F.softmax(out1, dim=-1)
                        flat = probs.view(1, -1, probs.shape[-1])
                        score_vals, _ = flat.max(dim=-1)  # [1, T]
                        t_star = int(score_vals.argmax(dim=1).item())
                        class_star = int(flat[0, t_star].argmax().item())
                        score = out1[0, t_star, class_star]
                        print("score: ", score)
                        score.backward(retain_graph=True)

                        # Compute CAMs pre/post sequence
                        assert hook.input_act is not None and hook.input_grad is not None
                        assert hook.output_act is not None and hook.output_grad is not None
                        cam_pre = _temporal_gradcam(hook.input_act, hook.input_grad)[0].detach().cpu().numpy()
                        cam_post = _temporal_gradcam(hook.output_act, hook.output_grad)[0].detach().cpu().numpy()

                        # Save npy
                        base = f"{str(utt_id).replace('/', '_')}_t{t_star}_c{class_star}"
                        np.save(os.path.join(vis_dir, f"{tag}_{rank:02d}_{base}_preseq_cam.npy"), cam_pre)
                        np.save(os.path.join(vis_dir, f"{tag}_{rank:02d}_{base}_postseq_cam.npy"), cam_post)

                        # Plot with label spans and predicted score
                        try:
                            import matplotlib.patches as mpatches
                            cam_pre_np = cam_pre
                            cam_post_np = cam_post
                            # predicted frame-level score from current forward
                            pred_score = (out2[0, :, 1] - out2[0, :, 0]).detach().cpu().numpy()
                            # align labels to CAM length
                            target_T = len(cam_post_np)
                            if len(y_np) != target_T:
                                src_idx = np.linspace(0, len(y_np) - 1, num=len(y_np))
                                tgt_idx = np.linspace(0, len(y_np) - 1, num=target_T)
                                y_np_vis = y_np[np.clip(np.round(tgt_idx).astype(int), 0, len(y_np) - 1)]
                            else:
                                y_np_vis = y_np
                            y_vis_line = y_np_vis.copy()
                            max_lab = y_vis_line.max() if y_vis_line.size > 0 else 1.0
                            if max_lab > 1:
                                y_vis_line = y_vis_line / (max_lab if max_lab != 0 else 1.0)

                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(cam_pre_np, label='Grad-CAM (pre-seq)', linewidth=2)
                            ax.plot(cam_post_np, label='Grad-CAM (post-seq)', linewidth=2)
                            ax.plot(pred_score, label='Pred score (pos-neg)', linewidth=1.2, alpha=0.85)
                            # EER threshold line for reference
                            ax.axhline(float(thresh_eer), color='gray', linestyle='--', linewidth=1.0, label=f'thresh={float(thresh_eer):.3f}')

                            unique_labels = np.unique(y_np_vis.astype(int))
                            cmap = plt.get_cmap('tab10')
                            label_to_color = {int(lab): cmap(i % 10) for i, lab in enumerate(unique_labels)}

                            T = len(y_np_vis)
                            start = 0
                            spans = []
                            for t in range(1, T + 1):
                                if t == T or y_np_vis[t] != y_np_vis[start]:
                                    lab = int(y_np_vis[start])
                                    spans.append((start, t, lab))
                                    start = t
                            for s, e, lab in spans:
                                ax.axvspan(s, e, facecolor=label_to_color[lab], alpha=0.15, linewidth=0)

                            ax.step(range(len(y_vis_line)), y_vis_line, where='post', alpha=0.5, linewidth=1.0, label='Label (scaled)')
                            label_patches = [mpatches.Patch(color=label_to_color[int(l)], alpha=0.3, label=f"Label={int(l)}") for l in unique_labels]
                            ax.set_title(f"{tag.upper()} | utt={utt_id} | err={diffs[idx]:.3f} | class={class_star} t*={t_star}")
                            ax.set_xlabel('Time index (segments)')
                            ax.set_ylabel('Importance / Label (scaled)')
                            ax.legend(loc='upper right', handles=ax.get_legend_handles_labels()[0] + label_patches)
                            out_png = os.path.join(vis_dir, f"{tag}_{rank:02d}_{base}_dual_cam.png")
                            fig.tight_layout()
                            fig.savefig(out_png, dpi=150)
                            plt.close(fig)
                        except Exception as e:
                            print(f"Plotting Grad-CAM failed for utt={utt_id}: {e}")
                    finally:
                        try:
                            fh.remove()
                            bh.remove()
                        except Exception:
                            pass

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

