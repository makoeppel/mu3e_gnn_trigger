import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class or binary classification.
    
    Args:
        alpha (float or list of floats): weighting factor for classes (optional)
        gamma (float): focusing parameter, usually 2.0
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha, 1-alpha])
            else:
                self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        """
        inputs: [N, C] logits (before softmax) for multi-class
                or [N] logits for binary classification
        targets: [N] integer class labels (0..C-1) for multi-class
                 or [N] 0/1 for binary classification
        """
        if inputs.dim() > 1 and inputs.size(1) > 1:
            # multi-class
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
            pt = torch.exp(-ce_loss)
        else:
            # binary
            inputs = inputs.view(-1)
            targets = targets.view(-1).float()
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            if self.alpha is not None:
                alpha_factor = targets * self.alpha[0] + (1-targets) * self.alpha[1]
                ce_loss = ce_loss * alpha_factor

        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
