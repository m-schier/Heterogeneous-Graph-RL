_current_ctx = None
_next_ctx_id = 0
_libsumo_ctx = None


class Context:
    def __init__(self, net_file, route_file, step_length=.1, mode='libsumo', auto_start_gui=True, config_file=None):
        global _next_ctx_id

        if mode == 'libsumo':
            self.__binary = ['sumo']
        elif mode == 'sumo':
            from sumolib import checkBinary
            self.__binary = [checkBinary('sumo')]
        elif mode == 'sumo-gui':
            from sumolib import checkBinary
            self.__binary = [checkBinary('sumo-gui')]
        else:
            raise ValueError("Bad mode: {}".format(mode))

        self.__id = _next_ctx_id
        _next_ctx_id += 1

        self.__net_file = net_file
        self.__route_file = route_file
        self.__step_length = step_length
        self.__mode = mode
        self.__is_open = False
        self.auto_start_gui = auto_start_gui
        self.__config_file = config_file
        self.__seed = None

        self.__debug('Created')

    def set_seed(self, seed):
        self.__seed = seed

    def get_seed(self):
        return self.__seed

    def open(self):
        global _current_ctx, _libsumo_ctx

        if self.__is_open:
            raise ValueError("Trying to open {}, which is already open".format(self))

        self.__debug('Opening')

        if _libsumo_ctx is not None and self.__mode == 'libsumo':
            raise ValueError("Trying to open libsumo Context {}, but already have a libsumo context".format(self))

        # if _current_ctx is not None:
        #     raise ValueError("Trying to open {}, but already have existing context {}".format(self, _current_ctx))

        args = self.__binary + ['--step-length', str(self.__step_length),
                   '--time-to-teleport', '-1',  # Never teleport congested vehicles
                   '--collision.mingap-factor', '0',  # Only actual physical collisions count as collisions
                   '--collision.check-junctions', 'true',  # Check junction collisions regarding vehicle width
                   '--collision.action', 'remove',  # Remove collided vehicles
                   '-n', self.__net_file]

        if self.__route_file is not None:
            args += ['-r', self.__route_file]

        if self.__config_file is not None:
            args += ['-c', self.__config_file]

        if self.auto_start_gui and self.__mode == 'sumo-gui':
            args += ['--start', '--quit-on-end']

        if self.__seed is not None:
            args += ['--seed', str(self.__seed)]

        try:
            if self.__mode == 'libsumo':
                import libsumo
                libsumo.start(args)
            else:
                import traci
                traci.start(args, label=self.get_identifier())
        except Exception as ex:
            raise RuntimeError("Failed to start SUMO with arguments '{}'".format(" ".join(args))) from ex
        _current_ctx = self
        self.__is_open = True

        if self.__mode == 'libsumo':
            _libsumo_ctx = self

    def __enter__(self):
        self.open()
        return self

    def close(self):
        global _current_ctx, _libsumo_ctx

        if not self.__is_open:
            # Calling close() on an unopen context has no effect
            return

        # if _current_ctx != self:
        #     raise ValueError("Trying to close {} while actually active is {}".format(self, _current_ctx))

        if self.__mode == 'libsumo':
            if _libsumo_ctx != self:
                raise ValueError("There should only be one libsumo context, but this one is different")

            import libsumo
            libsumo.close()
            _libsumo_ctx = None
        else:
            self.get_traci_module().close(wait=False)
            # import traci
            # traci.close(wait=False)
        self.__debug('Closed')
        _current_ctx = None
        self.__is_open = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return "Context(mode={}, id={})".format(self.__mode, self.__id)

    @property
    def net_file(self):
        return self.__net_file

    @staticmethod
    def get_current_context() -> 'Context':
        # The state caused so many issues here, remove it
        raise NotImplementedError
        # return _current_ctx

    def get_identifier(self):
        return "{}-{}".format(self.__mode, self.__id)

    def get_traci_module(self):
        if not self.__is_open:
            raise ValueError("Trying to get TraCI module for unopen {}".format(self))

        if self.__mode == 'libsumo':
            import libsumo
            return libsumo
        else:
            import traci
            return traci.getConnection(self.get_identifier())

    def __debug(self, action):
        from logging import debug
        debug("{} {}", action, self)
