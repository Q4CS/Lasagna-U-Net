import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class CEDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, n_classes=2, weight=None, softmax=True, include_background=True):
        super(CEDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight
        self.softmax = softmax
        if include_background:
            self.start_index = 0
        else:
            self.start_index = 1
        self.ce_loss = CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta

    def to_one_hot(self, input_tensor):
        # tensor: (N, C, H, W) or (N, H, W)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(dim=1)
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i * torch.ones_like(input_tensor))
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def dice_loss(self, inputs, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(inputs * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(inputs * inputs)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self.to_one_hot(target)

        # ce loss
        loss_ce = self.ce_loss(inputs, target)

        # dice loss
        if self.weight is None:
            self.weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.start_index, self.n_classes):
            dice = self.dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weight[i]

        loss_dice = loss / self.n_classes

        ce_dice_loss = self.alpha * loss_ce + self.beta * loss_dice
        return ce_dice_loss

