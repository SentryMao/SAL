import torch
from torchmetrics import Metric

# Additional imports for RangeEERMetric


class EERMetric(Metric):
    def __init__(self, percent=True):
        super().__init__(
            # compute_on_cpu=False,
            # sync_on_compute=False,
            # dist_sync_on_step=True
        )
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("indices", default=[], dist_reduce_fx="cat")
        self.percent = percent

    def update(self, preds: torch.Tensor, targets: torch.Tensor, indices=None):
        """Update state with predictions and targets."""
        self.y_pred.append(preds.detach())
        self.y_true.append(targets.detach())
        if indices is not None:
            self.indices.append(indices.detach())

    def _get_prediction(self):
        preds = torch.cat(self.y_pred).cpu().numpy()
        indices = torch.cat(self.indices).cpu().numpy()
        return preds, indices

    def compute(self):
        if not isinstance(self.y_pred, list):
            self.y_pred = [self.y_pred]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)

        # Sort predictions and corresponding labels. This is O(N log N).
        sorted_indices = torch.argsort(y_pred, descending=True)
        y_true_sorted = y_true[sorted_indices]

        # Calculate cumulative TPs and FPs using vectorized operations.
        tp = torch.cumsum(y_true_sorted, dim=0)
        fp = torch.cumsum(1 - y_true_sorted, dim=0)

        # Calculate total number of positive and negative samples
        pos_total = tp[-1]
        neg_total = fp[-1]

        # Handle the case where there are no positive or negative samples
        if pos_total == 0 or neg_total == 0:
            return (100.0, 0.0) if self.percent else (1.0, 0.0)

        # Calculate TPR and FPR for all thresholds at once. This is O(N).
        tpr = tp / pos_total
        fpr = fp / neg_total

        # Calculate EER
        abs_diff = torch.abs(fpr - (1 - tpr))
        eer_index = torch.argmin(abs_diff)
        eer = fpr[eer_index]
        thresh = y_pred[sorted_indices][eer_index]

        if self.percent:
            eer *= 100

        return eer.item(), thresh.item()


class F1Metric(Metric):
    def __init__(self, percent=True):
        super().__init__(
            # compute_on_cpu=False,
            # sync_on_compute=False,
            # dist_sync_on_step=True
        )
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.percent = percent

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update state with predictions and targets."""
        self.y_pred.append(preds.detach())
        self.y_true.append(targets.detach())

    def compute(self):
        """Compute the ACC and F1 score using PyTorch."""
        if not isinstance(self.y_pred, list):
            self.y_pred = [self.y_pred]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]

        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        y_pred = y_pred.argmax(dim=1)  # Convert logits to class labels

        # Calculate accuracy
        correct = (y_pred == y_true).sum().float()
        total = y_true.size(0)
        acc = correct / total if total > 0 else 0.0

        # Calculate F1 score
        tp = (y_pred * y_true).sum().float()
        fp = (y_pred * (1 - y_true)).sum().float()
        fn = ((1 - y_pred) * y_true).sum().float()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = 2 * (precision * recall) / (
                    precision + recall) if precision + recall > 0 else 0.0

        if self.percent:
            acc *= 100
            f1 *= 100

        return acc, f1


class CosThetaEERMetric(Metric):
    """EER metric for cos theta values from OCSoftmax."""

    def __init__(self, percent=True):
        super().__init__()
        self.add_state("cos_theta", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.percent = percent

    def update(self, cos_theta: torch.Tensor, targets: torch.Tensor):
        """Update state with cos theta values and targets."""
        self.cos_theta.append(cos_theta.detach())
        self.y_true.append(targets.detach())

    def compute(self):
        """Compute EER using cos theta values."""
        if not isinstance(self.cos_theta, list):
            self.cos_theta = [self.cos_theta]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]
        cos_theta = torch.cat(self.cos_theta)
        y_true = torch.cat(self.y_true)

        # Sort cos theta values and corresponding labels
        sorted_indices = torch.argsort(cos_theta, descending=True)
        cos_theta_sorted = cos_theta[sorted_indices]
        y_true_sorted = y_true[sorted_indices]

        # Calculate FPR and TPR
        thresholds, _ = torch.sort(torch.unique(cos_theta_sorted),
                                   descending=True)
        tpr = torch.zeros_like(thresholds, dtype=torch.float32)
        fpr = torch.zeros_like(thresholds, dtype=torch.float32)

        pos = (y_true_sorted == 1).sum().float()
        neg = (y_true_sorted == 0).sum().float()

        for i, thresh in enumerate(thresholds):
            predictions = (cos_theta_sorted >= thresh).float()
            tp = (predictions * y_true_sorted).sum().float()
            fp = (predictions * (1 - y_true_sorted)).sum().float()

            tpr[i] = tp / pos if pos > 0 else 0.0
            fpr[i] = fp / neg if neg > 0 else 0.0

        # Calculate EER
        abs_diff = torch.abs(fpr - (1 - tpr))
        eer_index = torch.argmin(abs_diff)
        eer = fpr[eer_index]
        thresh = thresholds[eer_index]

        if self.percent:
            eer *= 100

        return eer, thresh


class CosThetaAccMetric(Metric):
    """Accuracy metric for cos theta values from OCSoftmax."""

    def __init__(self, percent=True, threshold=0.0):
        super().__init__()
        self.add_state("cos_theta", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.percent = percent
        self.threshold = threshold

    def update(self, cos_theta: torch.Tensor, targets: torch.Tensor):
        """Update state with cos theta values and targets."""
        self.cos_theta.append(cos_theta.detach())
        self.y_true.append(targets.detach())

    def compute(self):
        """Compute accuracy using cos theta values."""
        if not isinstance(self.cos_theta, list):
            self.cos_theta = [self.cos_theta]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]

        cos_theta = torch.cat(self.cos_theta)
        y_true = torch.cat(self.y_true)

        # Use threshold to determine predictions
        predictions = (cos_theta >= self.threshold).float()

        # Calculate accuracy
        correct = (predictions == y_true).sum().float()
        total = y_true.size(0)
        acc = correct / total if total > 0 else 0.0

        # Calculate F1 score
        tp = (predictions * y_true).sum().float()
        fp = (predictions * (1 - y_true)).sum().float()
        fn = ((1 - predictions) * y_true).sum().float()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = 2 * (precision * recall) / (
                    precision + recall) if precision + recall > 0 else 0.0

        if self.percent:
            acc *= 100
            f1 *= 100

        return acc, f1


class EERMetricWithLabel(Metric):
    """EER metric with per-label statistics for labels 0-7."""

    def __init__(self, percent=True):
        super().__init__()
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("indices", default=[], dist_reduce_fx="cat")
        self.percent = percent

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               labels: torch.Tensor, indices=None):
        """Update state with predictions, targets, and labels."""
        self.y_pred.append(preds.detach())
        self.y_true.append(targets.detach())
        self.labels.append(labels.detach())
        if indices is not None:
            self.indices.append(indices.detach())

    def _get_prediction(self):
        preds = torch.cat(self.y_pred).cpu().numpy()
        indices = torch.cat(self.indices).cpu().numpy()
        return preds, indices

    def compute(self):
        if not isinstance(self.y_pred, list):
            self.y_pred = [self.y_pred]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]
        if not isinstance(self.labels, list):
            self.labels = [self.labels]

        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        labels = torch.cat(self.labels)

        # Calculate overall EER
        sorted_indices = torch.argsort(y_pred, descending=True)
        y_pred_sorted = y_pred[sorted_indices]
        y_true_sorted = y_true[sorted_indices]

        thresholds, _ = torch.sort(torch.unique(y_pred_sorted), descending=True)
        tpr = torch.zeros_like(thresholds, dtype=torch.float32)
        fpr = torch.zeros_like(thresholds, dtype=torch.float32)

        pos = (y_true_sorted == 1).sum().float()
        neg = (y_true_sorted == 0).sum().float()

        for i, thresh in enumerate(thresholds):
            predictions = (y_pred_sorted >= thresh).float()
            tp = (predictions * y_true_sorted).sum().float()
            fp = (predictions * (1 - y_true_sorted)).sum().float()

            tpr[i] = tp / pos if pos > 0 else 0.0
            fpr[i] = fp / neg if neg > 0 else 0.0

        abs_diff = torch.abs(fpr - (1 - tpr))
        eer_index = torch.argmin(abs_diff)
        eer = fpr[eer_index]
        thresh = thresholds[eer_index]

        if self.percent:
            eer *= 100

        # Calculate per-label statistics
        label_stats = {}
        for label_idx in range(8):
            label_mask = (labels == label_idx)
            if label_mask.sum() > 0:
                label_pred = y_pred[label_mask]
                label_true = y_true[label_mask]

                # Calculate EER for this label
                sorted_indices_label = torch.argsort(label_pred,
                                                     descending=True)
                label_pred_sorted = label_pred[sorted_indices_label]
                label_true_sorted = label_true[sorted_indices_label]

                thresholds_label, _ = torch.sort(
                    torch.unique(label_pred_sorted), descending=True)
                tpr_label = torch.zeros_like(thresholds_label,
                                             dtype=torch.float32)
                fpr_label = torch.zeros_like(thresholds_label,
                                             dtype=torch.float32)

                pos_label = (label_true_sorted == 1).sum().float()
                neg_label = (label_true_sorted == 0).sum().float()

                for i, thresh_label in enumerate(thresholds_label):
                    predictions_label = (
                                label_pred_sorted >= thresh_label).float()
                    tp_label = (
                                predictions_label * label_true_sorted).sum().float()
                    fp_label = (predictions_label * (
                                1 - label_true_sorted)).sum().float()

                    tpr_label[
                        i] = tp_label / pos_label if pos_label > 0 else 0.0
                    fpr_label[
                        i] = fp_label / neg_label if neg_label > 0 else 0.0

                abs_diff_label = torch.abs(fpr_label - (1 - tpr_label))
                if len(abs_diff_label) > 0:
                    eer_index_label = torch.argmin(abs_diff_label)
                    eer_label = fpr_label[eer_index_label]
                    if self.percent:
                        eer_label *= 100
                else:
                    eer_label = torch.tensor(0.0, device=y_pred.device)

                # Count samples
                total_samples = label_mask.sum().item()
                positive_samples = (label_true == 1).sum().item()
                negative_samples = (label_true == 0).sum().item()

                label_stats[label_idx] = {
                    'eer': eer_label.item(),
                    'total_samples': total_samples,
                    'positive_samples': positive_samples,
                    'negative_samples': negative_samples
                }
            else:
                label_stats[label_idx] = {
                    'eer': 0.0,
                    'total_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0
                }

        return eer, thresh, label_stats


class F1MetricWithLabel(Metric):
    """F1 metric with per-label statistics for labels 0-7."""

    def __init__(self, percent=True):
        super().__init__()
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.percent = percent

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               labels: torch.Tensor):
        """Update state with predictions, targets, and labels."""
        self.y_pred.append(preds.detach())
        self.y_true.append(targets.detach())
        self.labels.append(labels.detach())

    def compute(self):
        """Compute the ACC and F1 score with per-label statistics."""
        if not isinstance(self.y_pred, list):
            self.y_pred = [self.y_pred]
        if not isinstance(self.y_true, list):
            self.y_true = [self.y_true]
        if not isinstance(self.labels, list):
            self.labels = [self.labels]

        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        labels = torch.cat(self.labels)
        y_pred = y_pred.argmax(dim=1)  # Convert logits to class labels

        # Calculate overall accuracy and F1
        correct = (y_pred == y_true).sum().float()
        total = y_true.size(0)
        acc = correct / total if total > 0 else 0.0

        tp = (y_pred * y_true).sum().float()
        fp = (y_pred * (1 - y_true)).sum().float()
        fn = ((1 - y_pred) * y_true).sum().float()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = 2 * (precision * recall) / (
                    precision + recall) if precision + recall > 0 else 0.0

        if self.percent:
            acc *= 100
            f1 *= 100

        # Calculate per-label statistics
        label_stats = {}
        for label_idx in range(8):
            label_mask = (labels == label_idx)
            if label_mask.sum() > 0:
                label_pred = y_pred[label_mask]
                label_true = y_true[label_mask]

                # Calculate accuracy for this label
                label_correct = (label_pred == label_true).sum().float()
                label_total = label_true.size(0)
                label_acc = label_correct / label_total if label_total > 0 else 0.0

                # Calculate F1 for this label
                label_tp = (label_pred * label_true).sum().float()
                label_fp = (label_pred * (1 - label_true)).sum().float()
                label_fn = ((1 - label_pred) * label_true).sum().float()

                label_precision = label_tp / (
                            label_tp + label_fp) if label_tp + label_fp > 0 else 0.0
                label_recall = label_tp / (
                            label_tp + label_fn) if label_tp + label_fn > 0 else 0.0
                label_f1 = 2 * (label_precision * label_recall) / (
                            label_precision + label_recall) if label_precision + label_recall > 0 else 0.0

                if self.percent:
                    label_acc *= 100
                    label_f1 *= 100

                # Count samples
                total_samples = label_mask.sum().item()
                positive_samples = (label_true == 1).sum().item()
                negative_samples = (label_true == 0).sum().item()
                print('acc', label_acc)
                print('f1', label_f1)
                print('precision', label_precision)
                print('recall', label_recall)
                print('total_samples', total_samples)
                print('positive_samples', positive_samples)
                print('negative_samples', negative_samples)
                label_stats[label_idx] = {
                    'acc': label_acc.item(),
                    'total_samples': total_samples,
                    'positive_samples': positive_samples,
                    'negative_samples': negative_samples
                }
            else:
                label_stats[label_idx] = {
                    'acc': 0.0,
                    'total_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0
                }

        return acc, f1, label_stats
