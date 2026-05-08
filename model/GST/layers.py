import torch.nn as nn

# This was fixed by AI, go kill me understood. Oh nooo Ai SLOP noooooooooo
class FiLM_Layer(nn.Module):
    def __init__(self, hidden_dim, conditioning_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(conditioning_dim, hidden_dim * 2)

    def forward(self, x, condition):
        # x: [B, T, D]
        # condition: [B, C]

        gamma_beta = self.to_gamma_beta(condition)  # [B, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(1)  # [B, 1, D]
        beta = beta.unsqueeze(1)    # [B, 1, D]

        return gamma * x + beta

