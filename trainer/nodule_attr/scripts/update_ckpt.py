import argparse
import logging
from glob import glob
from pathlib import Path

import torch

from data_lake.constants import NUM_FOLD
from trainer.common.enums import ThresholdMode

LOGISTIC_STANDARD = "_logistic"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TARGET_ATTR = [
    "c_malignancy_logistic",
    "c_subtlety_logistic",
    "c_sphericity_logistic",
    "c_lobulation_logistic",
    "c_spiculation_logistic",
    "c_margin_logistic",
    "c_texture_logistic",
    "c_calcification_logistic",
    "c_internalStructure_logistic",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate ckpt")
    parser.add_argument("--model_num", type=int, default=0)
    args = parser.parse_args()
    logger.info("Merge checkpoints.")
    model_num = args.model_num

    for fold_idx in range(NUM_FOLD - 1):
        model_name_phase_1 = f"cls_all_model_{model_num}_val_fold{fold_idx}"
        model_name_phase_2 = f"cls_fine_model_{model_num}_val_fold{fold_idx}_*"
        base_model_path = f"/opt/challenge/trainer/nodule_attr/outputs/cls/baseline/{model_name_phase_1}/model.pth"
        fine_model_path = f"/opt/challenge/trainer/nodule_attr/outputs/cls/fine_tune/{model_name_phase_2}"
        save_model_path = f"/opt/challenge/trainer/nodule_attr/outputs/cls/baseline/{model_name_phase_1}/model_fine.pth"

        # check the number of models for fine
        search_path = Path(fine_model_path) / Path("model.pth")
        ckpt_paths = glob(str(search_path), recursive=True)
        assert len(ckpt_paths) == len(
            _TARGET_ATTR
        ), f"The number of checkpoint should be {len(_TARGET_ATTR)}, but got {len(ckpt_paths)}."

        # load base model
        checkpoint_source = torch.load(base_model_path)

        # merge param and threshold depending on the attrs
        for attr in _TARGET_ATTR:
            ckpt_candi_paths = [ckpt_path for ckpt_path in ckpt_paths if attr in ckpt_path]
            assert len(ckpt_candi_paths) == 1
            ckpt_path = ckpt_candi_paths[0]

            # dict_keys(['epoch', 'model', 'optimizer', 'threshold_c_subtlety_logistic'])
            _checkpoint = torch.load(ckpt_path)

            # update params related to a specific attr
            keys_to_merge = [param for param in _checkpoint["model"].keys() if attr in param]
            for key in keys_to_merge:
                checkpoint_source["model"][key].copy_(_checkpoint["model"][key])

            # update threshold
            if LOGISTIC_STANDARD in attr:
                checkpoint_source[f"threshold_{ThresholdMode.YOUDEN.value}_{attr}"] = _checkpoint[
                    f"threshold_{ThresholdMode.YOUDEN.value}_{attr}"
                ]

        # save merged model
        torch.save(checkpoint_source, save_model_path)


if __name__ == "__main__":
    main()
