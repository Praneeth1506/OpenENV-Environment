class DemoSafeSignalEnv(SafeSignalEnv):
    """
    Forced scenario for demo purposes.
    Overrides random transitions with a scripted story arc.
    """
    FORCED_STATES = [
        "SAFE", "SAFE", "SAFE", "SAFE", "SAFE",        # Days 1-5: normal
        "SAFE", "SAFE", "VULNERABLE", "VULNERABLE",     # Days 6-9: drift begins
        "VULNERABLE", "VULNERABLE", "AT_RISK",          # Days 10-12: escalating
        "AT_RISK", "AT_RISK",                           # Days 13-14: serious
        "VULNERABLE", "VULNERABLE",                     # Days 15-16: after intervention
        "VULNERABLE", "SAFE", "SAFE", "SAFE",           # Days 17-20: recovering
        "SAFE", "SAFE", "SAFE", "SAFE", "SAFE",        # Days 21-25: stable
        "SAFE", "SAFE", "SAFE", "SAFE", "SAFE",        # Days 26-30: resolved
    ]

    def __init__(self):
        super().__init__(archetype="target")
        self._day_index = 0

    def reset(self):
        self._day_index = 0
        state = super().reset()
        # Force initial hidden state
        self.child.hidden_state = self.FORCED_STATES[0]
        self.child.hidden_state_numeric = \
            ["SAFE","VULNERABLE","AT_RISK","IN_DANGER"].index(
                self.FORCED_STATES[0]
            )
        return state

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        # Override hidden state with scripted arc
        self._day_index += 1
        if self._day_index < len(self.FORCED_STATES):
            forced = self.FORCED_STATES[self._day_index]
            self.child.hidden_state = forced
            self.child.hidden_state_numeric = \
                ["SAFE","VULNERABLE","AT_RISK","IN_DANGER"].index(forced)
            info["hidden_state"] = forced
        return next_state, reward, done, info
