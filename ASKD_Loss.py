import torch
import torch.nn as nn
import torch.nn.functional as F

class ASKD(nn.Module):
    def __init__(self, threshold, AKD_alpha, SKD_alpha, temperature, ignore_index=-100, label_smoothing=0):
        super(ASKD, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kl_loss = nn.KLDivLoss(reduction="none").to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,label_smoothing=label_smoothing)
        self.threshold = threshold
        self.AKD_alpha = AKD_alpha
        self.SKD_alpha = SKD_alpha
        self.T = temperature


    def forward(self,logits, teacher_soft_labels, labels):
        teacher_soft_labels = teacher_soft_labels.detach()
        padding_mask = labels != self.ignore_index

        if not padding_mask.any():
            return logits[padding_mask].sum()
        
        if self.AKD_alpha > self.threshold:
            student_cn_loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            student_log_probs = F.log_softmax(logits / self.T, dim=-1)

            divergence = self.kl_loss(student_log_probs, teacher_soft_labels) * self.T ** 2

            divergence = divergence * padding_mask
            divergence = divergence.sum() / padding_mask.sum()

            loss = (divergence * self.AKD_alpha) + (student_cn_loss)
        # Self-knowledge Distillation 
        else: 
            student_log_probs = F.log_softmax(logits, dim=-1)
            
            adjusted_labels = labels.clone()
            adjusted_labels[~padding_mask] = 0

            one_hot_label = F.one_hot(
                adjusted_labels, num_classes=logits.size(-1)
            ).to(device=logits.device, dtype=student_log_probs.dtype)
            
            soft_target = ((1 - self.SKD_alpha) * one_hot_label) + (self.SKD_alpha * teacher_soft_labels)
            token_loss = (- soft_target * student_log_probs).sum(-1)
            loss = ((token_loss * padding_mask).sum() / padding_mask.sum())
        return loss        