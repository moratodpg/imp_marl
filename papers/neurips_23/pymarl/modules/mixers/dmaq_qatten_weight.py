import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl


class Qatten_Weight(nn.Module):
    def __init__(self, args):
        super(Qatten_Weight, self).__init__()

        self.name = 'qatten_weight'
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.n_actions = args.n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.n_head  # attention head num

        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):  # multi-head attention
            selector_nn = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim, bias=False))
            self.selector_extractors.append(selector_nn)  # query
            if self.args.nonlinear:  # add qs
                self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
            else:
                self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                              nn.ReLU(),
                                              nn.Linear(hypernet_embed, self.n_head))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                      th.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            if self.args.mask_dead:
                # actions: (episode_batch, episode_length - 1, agent_num, 1)
                actions = actions.reshape(-1, 1, self.n_agents)
                # actions: (batch_size, 1, agent_num)
                scaled_attend_logits[actions == 0] = -99999999  # action == 0 means the unit is dead
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = th.stack(head_attend_weights, dim=1)  # (batch_size, self.n_head, self.n_agents)
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        v = self.V(states).view(-1, 1)  # v: (bs, 1)
        # head_qs: [head_num, bs, 1]
        if self.args.weighted_head:
            w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
            w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)  # w_head: (bs, head_num, self.n_agents)
            head_attend *= w_head

        head_attend = th.sum(head_attend, dim=1)

        if not self.args.state_bias:
            v *= 0.

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1).mean()) for probs in head_attend_weights]

        return head_attend, v, attend_mag_regs, head_entropies
