import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UNetConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True):
        super(UNetConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size  # feature: height, width)
        print('Training Phase in UNetConvLSTM: {}, {}'.format(self.height, self.width))
        self.input_dim = input_dim  # input channel
        self.hidden_dim = hidden_dim  # output channel [16, 16, 16, 16, 16, 8]
        self.kernel_size = kernel_size  # kernel size  [[3, 3]*5]
        self.num_layers = num_layers  # Unet layer size: must be odd
        self.bias = bias

        cell_list = []
        self.down_num = (self.num_layers + 1) / 2

        for i in range(0, self.num_layers):
            scale = 2 ** i if i < self.down_num else 2 ** (self.num_layers - i - 1)
            cell_list.append(ConvLSTMCell(input_size=(int(self.height / scale), int(self.width / scale)),
                                          input_dim=self.input_dim[i],
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.deconv_0 = deConvGnReLU(
            16,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
        )
        self.deconv_1 = deConvGnReLU(
            16,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
        )
        self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)

    def forward(self, input_tensor, hidden_state=None, idx=0, process_sq=True):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if idx == 0:  # input the first layer of input image
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)

        cur_layer_input = input_tensor

        if process_sq:

            h0, c0 = hidden_state[0] = self.cell_list[0](input_tensor=cur_layer_input,
                                                         cur_state=hidden_state[0])

            h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
            h1, c1 = hidden_state[1] = self.cell_list[1](input_tensor=h0_1,
                                                         cur_state=hidden_state[1])

            h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)
            h2, c2 = hidden_state[2] = self.cell_list[2](input_tensor=h1_0,
                                                         cur_state=hidden_state[2])
            h2_0 = self.deconv_0(h2)  # auto reuse

            h2_1 = torch.cat([h2_0, h1], 1)
            h3, c3 = hidden_state[3] = self.cell_list[3](input_tensor=h2_1,
                                                         cur_state=hidden_state[3])
            h3_0 = self.deconv_1(h3)  # auto reuse
            h3_1 = torch.cat([h3_0, h0], 1)
            h4, c4 = hidden_state[4] = self.cell_list[4](input_tensor=h3_1,
                                                         cur_state=hidden_state[4])

            cost = self.conv_0(h4)  # auto reuse

            return cost, hidden_state
        else:
            for t in range(seq_len):
                h0, c0 = self.cell_list[0](input_tensor=cur_layer_input[:, t, :, :, :],
                                           cur_state=hidden_state[0])
                hidden_state[0] = [h0, c0]
                h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
                h1, c1 = self.cell_list[1](input_tensor=h0_1,
                                           cur_state=hidden_state[1])
                hidden_state[1] = [h1, c1]
                h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)
                h2, c2 = self.cell_list[2](input_tensor=h1_0,
                                           cur_state=hidden_state[2])
                hidden_state[2] = [h2, c2]
                h2_0 = self.deconv_0(h2)  # auto reuse

                h2_1 = torch.concat([h2_0, h1], 1)
                h3, c3 = self.cell_list[3](input_tensor=h2_1,
                                           cur_state=hidden_state[3])
                hidden_state[3] = [h3, c3]
                h3_0 = self.deconv_1(h3)  # auto reuse
                h3_1 = torch.concat([h3_0, h0], 1)
                h4, c4 = self.cell_list[4](input_tensor=h3_1,
                                           cur_state=hidden_state[4])
                hidden_state[4] = [h4, c4]

                cost = self.conv_0(h4)  # auto reuse
                cost = nn.Tanh(cost)
                # output cost
                layer_output_list.append(cost)

            prob_volume = torch.stack(layer_output_list, dim=1)

            return prob_volume

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class deConvGnReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 bias=True,
                 output_padding=1,
                 group_channel=8  # channel number in each group
                 ):
        super(deConvGnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                       output_padding=output_padding, stride=stride, bias=bias)
        self.group_channel = group_channel
        G = int(max(1, out_channels / self.group_channel))
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)
