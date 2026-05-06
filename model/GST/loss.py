from torch import nn

class TPSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, predicted_tokens, target):
        """
        calculate L1 loss function between predicted and target GST
        :param predicted_tokens: tensor shape of (batch_size, token_dim)
        :param target: tensor shape of (batch_size, token_dim)
        :return: L1 loss
        """
        return self.l1(predicted_tokens, target)
    


