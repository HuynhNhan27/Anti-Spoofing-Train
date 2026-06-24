import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOLoss(nn.Module):
    """
    Self-Distillation Loss (DINO Loss).
    Computes cross-entropy between student's softmax outputs and centered, sharpened teacher's outputs.
    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, epochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        # Centering vector registered as a buffer
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Schedule for teacher temperature warmup
        if epochs > 0:
            warmup_len = min(warmup_teacher_temp_epochs, epochs)
            self.teacher_temp_schedule = np.concatenate((
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_len),
                np.ones(max(0, epochs - warmup_len)) * teacher_temp
            ))
        else:
            self.teacher_temp_schedule = np.array([teacher_temp])

    def forward(self, student_output, teacher_output, epoch):
        """
        Calculates DINO loss.
        student_output shape: ((2 + N) * batch_size, out_dim)
        teacher_output shape: (2 * batch_size, out_dim)
        """
        student_out = student_output / self.student_temp
        
        # Get teacher temperature for the current epoch
        epoch_idx = min(epoch, len(self.teacher_temp_schedule) - 1)
        temp = self.teacher_temp_schedule[epoch_idx]
        
        # Center and sharpen teacher outputs (and stop gradients)
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach()

        total_loss = 0
        n_loss_terms = 0
        
        # Chunk outputs by batch size
        # student_output has (2 + N) chunks: 2 global + N local
        # teacher_output has 2 chunks: 2 global
        student_chunks = student_out.chunk(self.ncrops)
        teacher_chunks = teacher_out.chunk(2)
        
        for i, t_chunk in enumerate(teacher_chunks):
            for j, s_chunk in enumerate(student_chunks):
                if i == j:
                    # Skip matching global crop with itself
                    continue
                # Cross-entropy loss: -P2 * log P1
                loss = torch.sum(-t_chunk * F.log_softmax(s_chunk, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        
        # Update the teacher output center
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update the running center of teacher outputs using EMA.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # Check if distributed training is used (standard DataParallel does not trigger this)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        batch_center = batch_center / (len(teacher_output) * world_size)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
