import torch
import torch.nn as nn


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.weight = torch.tensor(weight) if weight is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.weight,
                                             reduction='none')

    def forward(self, input, target, mask=None):
        loss = self.criterion(input, target)
        # print(input.shape, target.shape, loss.shape) (batch_size, num_classes, seq_len), (batch_size, seq_len), (batch_size, seq_len)
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        return loss


class MaskedMultiClassCrossEntropyLoss(nn.Module):
    """
    Masked CrossEntropyLoss for multi-class boundary labels (4 classes: 0,1,2,3)
    - 0: middle positions
    - 1: start positions  
    - 2: end positions
    - 3: single-frame segments (start=end)
    """

    def __init__(self, weight=None, num_classes=4):
        super(MaskedMultiClassCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = torch.tensor(weight) if weight is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.weight,
                                             reduction='none')

    def forward(self, input, target, mask=None):
        """
        Args:
            input: (batch_size, num_classes, seq_len) or (batch_size, seq_len, num_classes)
            target: (batch_size, seq_len) with values in [0, 1, 2, 3]
            mask: (batch_size, seq_len) or None
        """
        # Ensure input is in correct format (batch_size, num_classes, seq_len)
        if input.dim() == 3 and input.size(-1) == self.num_classes:
            # (batch_size, seq_len, num_classes) -> (batch_size, num_classes, seq_len)
            input = input.transpose(1, 2)

        loss = self.criterion(input, target)

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        return loss


class AdaptiveWeightedMultiClassCrossEntropyLoss(nn.Module):
    """
    Adaptive Weighted Multi-Class CrossEntropyLoss with dynamic weight adjustment based on loss distribution
    
    This loss function extends MaskedMultiClassCrossEntropyLoss by:
    1. Accepting an additional adaptive_target tensor with 8 classes
    2. Computing loss distribution across these 8 classes
    3. Adjusting weights based on loss proportion (higher loss proportion = higher weight)
    """

    def __init__(self, weight=None, num_classes=4, num_adaptive_classes=8,
                 weight_alpha=1.0):
        super(AdaptiveWeightedMultiClassCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.num_adaptive_classes = num_adaptive_classes
        self.weight_alpha = weight_alpha
        self.weight = torch.tensor(weight) if weight is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.weight,
                                             reduction='none')

    def forward(self, input, target, adaptive_target, mask=None):
        """
        Args:
            input: (batch_size, num_classes, seq_len) or (batch_size, seq_len, num_classes)
            target: (batch_size, seq_len) with values in [0, 1, 2, 3]
            adaptive_target: (batch_size, seq_len) with values in [0, 1, ..., 7] for 8 classes
            mask: (batch_size, seq_len) or None
        """
        # Ensure input is in correct format (batch_size, num_classes, seq_len)
        if input.dim() == 3 and input.size(-1) == self.num_classes:
            # (batch_size, seq_len, num_classes) -> (batch_size, num_classes, seq_len)
            input = input.transpose(1, 2)

        # Compute base loss
        loss = self.criterion(input, target)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            effective_mask = mask
        else:
            effective_mask = torch.ones_like(loss)

        # Compute adaptive weights based on loss distribution across adaptive_target classes
        adaptive_weights = self._compute_adaptive_weights(loss, adaptive_target,
                                                          effective_mask)

        # Apply adaptive weights to loss
        weighted_loss = loss * adaptive_weights

        # Compute final loss
        if mask is not None:
            final_loss = weighted_loss.sum() / mask.sum()
        else:
            final_loss = weighted_loss.sum() / loss.numel()

        return final_loss

    def _compute_adaptive_weights(self, loss, adaptive_target, mask):
        """
        Compute adaptive weights based on loss distribution across adaptive_target classes
        
        Args:
            loss: (batch_size, seq_len) - per-position loss values
            adaptive_target: (batch_size, seq_len) - class labels for adaptive weighting
            mask: (batch_size, seq_len) - effective mask for valid positions
            
        Returns:
            adaptive_weights: (batch_size, seq_len) - weights for each position
        """
        batch_size, seq_len = loss.shape
        device = loss.device

        # Initialize adaptive weights
        adaptive_weights = torch.ones_like(loss)

        # Compute loss statistics for each adaptive class
        class_losses = []
        class_counts = []

        for class_idx in range(self.num_adaptive_classes):
            # Create mask for current class
            class_mask = (adaptive_target == class_idx) & (mask > 0)

            if class_mask.sum() > 0:
                # Compute average loss for this class
                class_loss = (loss * class_mask).sum() / class_mask.sum()
                class_count = class_mask.sum()
            else:
                # If no samples for this class, use mean loss
                class_loss = loss.mean()
                class_count = 1

            class_losses.append(class_loss)
            class_counts.append(class_count)

        # Convert to tensors
        class_losses = torch.stack(class_losses)  # (num_adaptive_classes,)
        class_counts = torch.tensor(class_counts, device=device,
                                    dtype=torch.float)  # (num_adaptive_classes,)

        # Compute loss proportions (normalized by class frequency)
        total_weighted_loss = (class_losses * class_counts).sum()
        if total_weighted_loss > 0:
            loss_proportions = (
                                           class_losses * class_counts) / total_weighted_loss
        else:
            loss_proportions = torch.ones_like(
                class_losses) / self.num_adaptive_classes

        # Compute adaptive weights based on loss proportions
        # Higher loss proportion = higher weight
        adaptive_class_weights = 1.0 + self.weight_alpha * loss_proportions

        # Apply class weights to each position
        for class_idx in range(self.num_adaptive_classes):
            class_mask = adaptive_target == class_idx
            adaptive_weights[class_mask] = adaptive_class_weights[class_idx]

        return adaptive_weights


if __name__ == '__main__':
    # Test original loss
    preds = torch.randn(1, 4, 2)  # (bs, frame, class)
    labels = torch.randint(0, 2, (1, 4))  # (bs, frame)
    mask = torch.randint(0, 2, (1, 4)).float()  # (bs, frame)
    criterion = MaskedCrossEntropyLoss()
    loss = criterion(preds.transpose(1, 2), labels, mask)
    print(f"Original Loss: {loss.item()}")

    # Test new multi-class loss
    preds_multi = torch.randn(1, 4, 4)  # (bs, frame, 4_classes)
    labels_multi = torch.randint(0, 4, (1, 4))  # (bs, frame) with values 0-3
    mask_multi = torch.randint(0, 2, (1, 4)).float()  # (bs, frame)
    criterion_multi = MaskedMultiClassCrossEntropyLoss()
    loss_multi = criterion_multi(preds_multi.transpose(1, 2), labels_multi,
                                 mask_multi)
    print(f"Multi-class Loss: {loss_multi.item()}")

    # Test adaptive weighted loss
    preds_adaptive = torch.randn(2, 6, 4)  # (bs, seq_len, 4_classes)
    labels_adaptive = torch.randint(0, 4,
                                    (2, 6))  # (bs, seq_len) with values 0-3
    adaptive_targets = torch.randint(0, 8,
                                     (2, 6))  # (bs, seq_len) with values 0-7
    mask_adaptive = torch.randint(0, 2, (2, 6)).float()  # (bs, seq_len)

    criterion_adaptive = AdaptiveWeightedMultiClassCrossEntropyLoss(
        weight_alpha=2.0)
    loss_adaptive = criterion_adaptive(preds_adaptive, labels_adaptive,
                                       adaptive_targets, mask_adaptive)
    print(f"Adaptive Weighted Loss: {loss_adaptive.item()}")
