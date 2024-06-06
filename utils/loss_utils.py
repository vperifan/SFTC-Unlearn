import torch
import torch.nn.functional as F


def kl_loss(model_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1., distill: bool = False):
    teacher_output_softmax = F.softmax(teacher_logits / temperature, dim=1)

    output_log_softmax = F.log_softmax(model_logits / temperature, dim=1)

    if distill:
        return F.kl_div(output_log_softmax, teacher_output_softmax, reduction='sum') * (temperature ** 2) / \
            model_logits.shape[0]

    return F.kl_div(output_log_softmax, teacher_output_softmax, reduction='batchmean')


def js_loss(model_logits, teacher_logits, temperature=1.):
    teacher_output_softmax = F.softmax(teacher_logits / temperature, dim=1)
    teacher_output_log_softmax = F.log_softmax(teacher_logits / temperature, dim=1)

    output_softmax = F.softmax(model_logits / temperature, dim=1)
    output_log_softmax = F.log_softmax(model_logits / temperature, dim=1)

    m = (teacher_output_softmax + output_softmax) / 2

    kl_div = F.kl_div(output_log_softmax, m, reduction='batchmean')
    kl_div2 = F.kl_div(teacher_output_log_softmax, m, reduction='batchmean')
    return 0.5 * kl_div + 0.5 * kl_div2


def custom_kl_loss(
        teacher_logits: torch.Tensor,
        dummy_logits: torch.Tensor,
        student_logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        kl_temperature: float = 1.,
) -> torch.Tensor:
    """Calculates a custom KL divergence loss.
    'Chundawat, V. S., Tarun, A. K., Mandal, M., & Kankanhalli, M. (2023, June).
    Can bad teaching induce forgetting? Unlearning in deep networks using an incompetent teacher.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7210-7217).'

    Args:
        teacher_logits (torch.Tensor): Teacher model logits.
        dummy_logits (torch.Tensor): Dummy model logits.
        student_logits (torch.Tensor): Student model logits.
        pseudo_labels (torch.Tensor): Pseudo labels indicating the retaining samples.
        kl_temperature (float): Temperature parameter for softmax scaling.

    Returns:
        torch.Tensor: The custom kl divergence loss
    """
    pseudo_labels = torch.unsqueeze(pseudo_labels, dim=1)

    teacher_output_softmax = F.softmax(teacher_logits / kl_temperature, dim=1)
    dummy_output_softmax = F.softmax(dummy_logits / kl_temperature, dim=1)

    output_log_softmax = F.log_softmax(student_logits / kl_temperature, dim=1)

    # Pseudo label 1 for forget samples and 0 for retain samples.
    out = pseudo_labels * dummy_output_softmax + (1 - pseudo_labels) * teacher_output_softmax
    kl_div = F.kl_div(output_log_softmax, out, reduction='batchmean')

    return kl_div


class SelectiveCrossEntropyLoss(torch.nn.Module):
    """
    A selective Cross Entropy Loss that considers only the samples with the specified pseudo label.
    By default, only the loss is calculated on the samples belonging to the retain set.
    """

    def __init__(self, reduction='mean', weight=None):
        super(SelectiveCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets, pseudo_labels, target_pseudo_label: int = 0):
        mask = pseudo_labels.eq(target_pseudo_label)
        inputs = inputs[mask]
        targets = targets[mask]
        if inputs.shape[0] == 0:  # check if there are any samples to compute loss
            return torch.tensor(0.0, requires_grad=True).to(inputs.device)

        loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)

        return loss
