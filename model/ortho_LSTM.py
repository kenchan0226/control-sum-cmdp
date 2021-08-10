import torch
import torch.nn as nn


class LinearWithTwoInput(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(LinearWithTwoInput, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, bias=True)
        self.U = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x):
        d, h = x
        return self.W(d) + self.U(h)


class OrthoLSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(OrthoLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.i_layer = nn.Sequential(LinearWithTwoInput(hidden_size, input_size), nn.Sigmoid())
        self.f_layer = nn.Sequential(LinearWithTwoInput(hidden_size, input_size), nn.Sigmoid())
        self.o_layer = nn.Sequential(LinearWithTwoInput(hidden_size, input_size), nn.Sigmoid())
        self.c_hat_layer = nn.Sequential(LinearWithTwoInput(hidden_size, input_size), nn.Tanh())
        self.g_layer = nn.Sequential(LinearWithTwoInput(hidden_size, input_size), nn.Sigmoid())

    def forward(self, d_t, h_t_minus_1, c_t_minus_1):
        """
        :param d_t: input attn context[batch_size, input_size]
        :param c_t_minus_1: [batch_size, hidden_size]
        :param h_t_minus_1: [batch_size, hidden_size]
        :return:
        """
        batch_size = d_t.size(0)
        i_t = self.i_layer((d_t, h_t_minus_1))  # [batch, hidden_size]
        f_t = self.f_layer((d_t, h_t_minus_1))  # [batch, hidden_size]
        o_t = self.o_layer((d_t, h_t_minus_1))  # [batch, hidden_size]
        c_hat_t = self.c_hat_layer((d_t, h_t_minus_1))  # [batch, hidden_size]
        g_t = self.g_layer((d_t, h_t_minus_1))  # [batch, hidden_size]
        c_t = i_t * c_hat_t + f_t * c_t_minus_1  # [batch, hidden_size]
        # [batch, 1, 1]
        fraction_term = torch.bmm( c_t.view(batch_size, 1, self.hidden_size), c_t_minus_1.view(batch_size, self.hidden_size, 1) ) \
                        / torch.bmm( c_t_minus_1.view(batch_size, 1, self.hidden_size), c_t_minus_1.view(batch_size, self.hidden_size, 1) )
        fraction_term = fraction_term.view(batch_size, 1).expand(batch_size, self.hidden_size)  # [batch, hidden_size]
        c_div_t = c_t - g_t * fraction_term * c_t_minus_1
        h_t = o_t * torch.tanh(c_div_t)
        return h_t, c_t


if __name__ == "__main__":
    hidden_size = 300
    batch_size = 4
    cell = OrthoLSTMCell(hidden_size=hidden_size, input_size=hidden_size)
    d_t = torch.randn(batch_size, hidden_size)
    h = torch.zeros(batch_size, hidden_size)
    c = torch.zeros(batch_size, hidden_size)
    h_t, c_t = cell(d_t, h, c)
    print(h_t.size())
    print(c_t.size())
