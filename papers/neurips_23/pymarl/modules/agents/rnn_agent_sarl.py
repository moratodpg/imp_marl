import torch.nn as nn


class RNNAgentSARL(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentSARL, self).__init__()
        self.args = args

        fc1_list = args.agent_fc1

        fc1_list.append(args.rnn_hidden_dim)
        fc1_list = [input_shape] + fc1_list
        self.fc1 = nn.Sequential()
        for i in range(1, len(fc1_list)):
            self.fc1.add_module(
                "lin" + str(i - 1), nn.Linear(fc1_list[i - 1], fc1_list[i])
            )
            self.fc1.add_module("Relu" + str(i - 1), nn.ReLU())

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        fc2_list = args.agent_fc2
        fc2_list = [args.rnn_hidden_dim] + fc2_list
        fc2_list.append(args.n_actions)
        self.fc2 = nn.Sequential()
        for i in range(1, len(fc2_list)):
            self.fc2.add_module(
                "fc2_lin" + str(i - 1), nn.Linear(fc2_list[i - 1], fc2_list[i])
            )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1[-2].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = self.fc1(inputs)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
