from torch import nn

class TPSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, predicted_tokens, target):
        """
        calculate L1 loss function between predicted and target GST
        :param predicted_tokens: tensor shape of (batch_size, token_dim)
        :param target: tensor shape of (batch_size, token_dim)
        :return: L1 loss
        """
        l1_loss = self.l1(predicted_tokens, target)
        cosine_loss = 1 - self.cosine(predicted_tokens, target).mean()


        loss = 0.3 * l1_loss + 0.7 * cosine_loss
        return loss #self.l1(predicted_tokens, target)
    


