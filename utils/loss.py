import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target, smooth=1e-5):
        # BCE loss (expects logits)
        bce_loss = self.bce(pred, target)

        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = torch.sum(pred_sigmoid * target)
        dice_loss = 1 - (2. * intersection + smooth) / (
            torch.sum(pred_sigmoid) + torch.sum(target) + smooth
        )

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
    
dice_bce_loss = DiceBCELoss(bce_weight=1.0, dice_weight=1.0)
