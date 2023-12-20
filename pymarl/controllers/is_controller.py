from controllers import BasicMAC


class ISMAC(BasicMAC):
    """
    Importance-Samping Controller.

    Same as Basic Controller but retrieve whole network output to compute
    IS ratio in the learner.

    """

    def __init__(self, scheme, groups, args):
        super(ISMAC, self).__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions, agent_outputs
