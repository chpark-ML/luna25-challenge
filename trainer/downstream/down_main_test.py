import hydra

from trainer.downstream.main import main

TEST_OVERRIDE_CONFIG = ["+debug=True", "+work_dir=foo", "+print_config=False", "loader.batch_size=4"]


def test_main():
    with hydra.initialize_config_module(config_module="trainer.downstream.configs", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=TEST_OVERRIDE_CONFIG)
        assert main(cfg) > 0  # If we can train the model deterministically, we can assert the value


def test_main_fold_10():
    with hydra.initialize_config_module(config_module="trainer.downstream.configs", version_base=None):
        cfg = hydra.compose(config_name="config",
                            overrides=TEST_OVERRIDE_CONFIG + ["loader.dataset.dataset_infos.luna25.fold_key=fold_10"])
        assert main(cfg) > 0  # If we can train the model deterministically, we can assert the value
