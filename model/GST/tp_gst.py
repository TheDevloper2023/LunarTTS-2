import torch
from torch import nn

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class TPSE(nn.Module):
    """
    Text-Predicting Style Embedding
    """
    def __init__(self, hparams):
        super().__init__()
        self.hidden_state_dim = hparams["tpgst"]["GST"]["tpse_gru_hidden_state_dim"]
        self.encoder_embedding_dim = hparams["transformer"]["encoder_hidden"] + \
                                    (hparams["tpgst"]["BERT"]["bert_encoder_dim"])
        self.fc_layers = hparams["tpgst"]["GST"]["tpse_fc_layers"]
        self.fc_layers_dim = hparams["tpgst"]["GST"]["tpse_fc_layer_dim"]
        self.token_dim = hparams["gst"]["token_size"]

        self.gru = nn.GRU(input_size=self.encoder_embedding_dim,
                          hidden_size=self.hidden_state_dim,
                          num_layers=1,
                          batch_first=True)

        self.fc_layers_model = None
        if self.fc_layers < 1:
            raise ValueError('hparams.fc_layers must be 1 or greater')
        elif self.fc_layers == 1:
            self.fc_layers_model = nn.Sequential(LinearNorm(self.hidden_state_dim, self.token_dim),
                                                 nn.Tanh())
        else:
            fc_layers_list = []
            # input layer
            fc_layers_list.append(LinearNorm(self.hidden_state_dim, self.fc_layers_dim))
            fc_layers_list.append(nn.ReLU())
            # hidden layers
            for i in range(self.fc_layers - 2):
                fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.fc_layers_dim))
                fc_layers_list.append(nn.ReLU())
            # output layer
            fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.token_dim))
            fc_layers_list.append(nn.Tanh())

            self.fc_layers_model = nn.Sequential(*fc_layers_list)

    def forward(self, inputs):
        """
        forwarding through the model layers
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: style token tensor shape of (batch_size, token_dim)
        """
        self.gru.flatten_parameters()
        _, hidden_state_n = self.gru(inputs)
        # hidden_state_n - tensor shape of (1, batch_size, hidden_state_dim)
        # bring to shape (batch_size, hidden_state_dim)
        hidden_state_n = hidden_state_n.squeeze(dim=0)
        fc_output = self.fc_layers_model(hidden_state_n)

        return fc_output
    



#Might Remove Later on, I am keeping it as alternitive to TPSE
#It trains faster and thats it
#Atm it is not conected to anything in the actual model code
class TPSELinear(nn.Module):
    """
    Text-Predicting Style Embedding (without rnn layer)
    """
    def __init__(self, hparams):
        super().__init__()
        self.encoder_embedding_dim = hparams["tpgst"]["GST"]["hparams.encoder_embedding_dim"] + \
                                    (hparams.bert_encoder_dim if hparams.tp_gst_use_bert else 0)
        self.fc_layers = hparams["tpgst"]["GST"]["tpse_fc_layers"]
        self.fc_layers_dim = hparams["tpgst"]["GST"]["tpse_fc_layer_dim"]
        self.token_dim = hparams["tpgst"]["GST"]["token_embedding_size"]

        self.fc_layers_model = None
        if self.fc_layers < 1:
            raise ValueError('hparams.fc_layers must be 1 or greater')
        elif self.fc_layers == 1:
            self.fc_layers_model = nn.Sequential(LinearNorm(self.encoder_embedding_dim, self.token_dim),
                                                 nn.Tanh())
        else:
            fc_layers_list = []
            # input layer
            fc_layers_list.append(LinearNorm(self.encoder_embedding_dim, self.fc_layers_dim))
            fc_layers_list.append(nn.ReLU())
            # hidden layers
            for i in range(self.fc_layers - 2):
                fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.fc_layers_dim))
                fc_layers_list.append(nn.ReLU())
            # output layer
            fc_layers_list.append(LinearNorm(self.fc_layers_dim, self.token_dim))
            fc_layers_list.append(nn.Tanh())

            self.fc_layers_model = nn.Sequential(*fc_layers_list)

    def forward(self, inputs):
        """
        forwarding through the model layers
        :param inputs: encoder output shape of (batch_size, max_seq_len, embedding_dim)
        :return: style token tensor shape of (batch_size, token_dim)
        """

        fc_output = self.fc_layers_model(inputs)

        return fc_output