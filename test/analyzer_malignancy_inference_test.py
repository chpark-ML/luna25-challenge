from pathlib import Path

import hydra

from analyzer.malignancy.main_inference import main

TEST_OVERRIDE_CONFIG = ["sanity_check=True", "run_name=test"]


def test_main_inference():
    # Result file path
    result_file = Path("./result_test.csv")

    # Remove the previous result file if it exists
    if result_file.exists():
        result_file.unlink()

    with hydra.initialize_config_module(config_module="analyzer.malignancy.configs", version_base=None):
        cfg = hydra.compose(config_name="config_inference", overrides=TEST_OVERRIDE_CONFIG)

        # Run the inference main function
        main(cfg)
        assert result_file.exists(), f"Expected file not found: {result_file}"
