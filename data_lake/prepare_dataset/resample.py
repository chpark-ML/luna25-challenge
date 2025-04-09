import argparse
import multiprocessing as mp
import os
from glob import glob
from pathlib import Path

import h5py
from tqdm import tqdm

from data_lake.constants import DEFAULT_RESAMPLED_SPACING, DatasetKey, H5DataKey, LUNA25Dir, MetaDataKey
from data_lake.utils.itk_to_npy import itk_image_to_numpy_image
from data_lake.utils.resample_image import resample_image


def export_to_h5(output_path, volume, origin, spacing, transform, resampled, resampled_spacing):
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset(H5DataKey.image, data=volume)
        h5f.attrs[H5DataKey.origin] = origin
        h5f.attrs[H5DataKey.spacing] = spacing
        h5f.attrs[H5DataKey.transform] = transform

        h5f.create_dataset(H5DataKey.resampled_image, data=resampled)
        h5f.attrs[H5DataKey.resampled_spacing] = resampled_spacing


def process_single_file(args):
    input_path_str, output_dir_str = args
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)

    try:
        numpy_image, header = itk_image_to_numpy_image(input_path, mode=input_path.suffix)
        origin = header[MetaDataKey.origin]
        spacing = header[MetaDataKey.spacing]
        transform = header[MetaDataKey.transform]

        resampled = resample_image(numpy_image, spacing, DEFAULT_RESAMPLED_SPACING)

        output_path = output_dir / (input_path.stem + ".h5")
        export_to_h5(output_path, numpy_image, origin, spacing, transform, resampled, DEFAULT_RESAMPLED_SPACING)

        return f"Saved: {output_path}"
    except Exception as e:
        return f"Failed: {input_path} - {e}"


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by exporting images to h5 files.")
    parser.add_argument("--dataset", default=DatasetKey.luna25, type=str, choices=[DatasetKey.luna25])
    parser.add_argument("--num_shards", type=int, default=16, help="Number of parallel workers.")
    args = parser.parse_args()

    if args.dataset == DatasetKey.luna25:
        image_dir = Path(LUNA25Dir.image_dir)
        output_dir = Path(LUNA25Dir.output_dir)
        input_path_list = glob(str(image_dir / "*.mha"))

        assert len(input_path_list) == 4069
        stems = [Path(p).stem for p in input_path_list]
        if len(stems) == len(set(stems)):
            print("All stem values are unique.")
        else:
            print("There are duplicate stem values.")
    else:
        raise NotImplementedError

    os.makedirs(output_dir, exist_ok=True)
    num_shards = args.num_shards if args.num_shards else os.cpu_count()
    task_args = [(path, str(output_dir)) for path in input_path_list]
    print(f"Processing {len(input_path_list)} files with {args.num_shards} shards...")

    with mp.Pool(num_shards) as pool:
        for result in tqdm(pool.imap_unordered(process_single_file, task_args), total=len(task_args)):
            print(result)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # for compatibility across platforms
    main()
