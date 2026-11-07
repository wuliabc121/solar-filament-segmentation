import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


def get_loss():
    dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    def loss_fn(y_pred, y_true):
        return dice_loss(y_pred, y_true) + bce_loss(y_pred, y_true)
    return loss_fn
