import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-7):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(1)
    target = target.float().flatten(1)

    intersection = (probabilities * target).sum(dim=1)

    dice = (
        2.0 * intersection + eps
    ) / (
        probabilities.sum(dim=1)
        + target.sum(dim=1)
        + eps
    )

    return 1.0 - dice.mean()


def combined_loss(logits, target):
    target = target.float()

    # Give more importance to positive oil-spill pixels.
    positive_pixels = target.sum()
    total_pixels = target.numel()

    positive_fraction = positive_pixels / max(total_pixels, 1)

    # Keep the weight controlled so training does not become unstable.
    pos_weight_value = torch.clamp(
        0.5 / (positive_fraction + 1e-6),
        min=1.0,
        max=5.0
    )

    pos_weight = torch.tensor(
        [pos_weight_value.item()],
        device=logits.device,
        dtype=logits.dtype
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight
    )

    dice = dice_loss(logits, target)

    return bce + dice