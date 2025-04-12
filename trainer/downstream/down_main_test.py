import hydra

from trainer.downstream.main import main

TEST_OVERRIDE_CONFIG = [
    "+debug=True",
    "+work_dir=foo",
    "+print_config=False",
    "inputs.batch_size=4"
]


def test_main():
    with hydra.initialize_config_module(
        config_module="trainer.downstream.configs", version_base=None
    ):
        cfg = hydra.compose(config_name="config", overrides=TEST_OVERRIDE_CONFIG)
        assert main(cfg) > 0  # If we can train the model deterministically, we can assert the value
