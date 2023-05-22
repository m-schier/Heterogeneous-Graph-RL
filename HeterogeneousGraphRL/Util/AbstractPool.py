class AbstractEnvironmentPool:
    def async_reset(self):
        """
        Reset all environment asynchronously
        """
        raise NotImplementedError

    def spool(self, sync=False):
        """
        Query ready environments
        :param sync: If `True`, force waiting on all environments
        :return: Tuple of (list of new states, list of transitions), where both are not necessarily the same length.
        A transition is a tuple of (state, act, next_state, reward, done, *extra)
        """
        raise NotImplementedError

    def async_step(self, actions):
        """
        Asynchronously step the environment
        :param actions: List of actions to apply to each state, must have the order of states as previously returned by
        `spool()`
        """
        raise NotImplementedError

    def close(self):
        """
        Close any resources held
        """
        pass
