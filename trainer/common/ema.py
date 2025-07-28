class ExponentialMovingAverage:
    def __init__(self, enabled: bool = True, decay: float = 0.99):
        self.enabled = enabled
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialized = False

    def register(self, model):
        if not self.enabled:
            return
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._initialized = True

    def update(self, model):
        if not self.enabled:
            return
        if not self._initialized:
            raise RuntimeError("EMA is not initialized. Call register(model) first.")
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        if not self.enabled:
            return
        if not self._initialized:
            raise RuntimeError("EMA is not initialized. Call register(model) first.")
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        if not self.enabled:
            return
        if not self._initialized:
            raise RuntimeError("EMA is not initialized. Call register(model) first.")
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
