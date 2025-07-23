class SchedulerTool:
    """
    The way it is called depends on the type of scheduler, so this tool is used for handling.
    """

    def __init__(self, scheduler):
        # set scheduler
        self.scheduler = scheduler

        # get scheduler name
        self.scheduler_name = scheduler.__class__.__name__

        # This tool has been implemented considering the following scheduler instances.
        assert self.scheduler_name in [
            "NoneType",  # no scheduler
            "OneCycleLR",
            "CosineAnnealingWarmUpRestarts",
            "ExponentialLR",
            "CosineAnnealingLR",
            "ReduceLROnPlateau",
        ]

        # set model_mode between "epoch", "step", and "none".
        if self.scheduler_name == "NoneType":
            self.scheduler_mode = "none"
        elif self.scheduler_name == "OneCycleLR":
            self.scheduler_mode = "step"
        elif self.scheduler_name == "ReduceLROnPlateau":
            self.scheduler_mode = "epoch_val"
        else:
            self.scheduler_mode = "epoch"

    def step(self, mode="epoch", *args):
        if self.scheduler_mode == mode:
            self.scheduler.step(args[0]) if args else self.scheduler.step()
