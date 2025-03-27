import collections
import errno
import os
import shlex
import shutil
import subprocess
import tempfile
import warnings
from functools import partial
from glob import glob
from multiprocessing import Pool

import numpy as np
import pydicom as dicom
import SimpleITK as sitk


class DicomLoadException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


_LOADER_PARAM_NAMES = collections.namedtuple(
    "LoaderParams",
    [
        "delete_uncompressed",
        "kernel_priority",
        "uncompress_processor",
        "patience_for_inconsistency",
        "read_dcm_only",
    ],
)
_LOADER_PARAMS = _LOADER_PARAM_NAMES(
    read_dcm_only=False,
    delete_uncompressed=True,
    uncompress_processor=8,
    patience_for_inconsistency=3,
    kernel_priority={
        "CHST": 2,  # GE, for mediastinum and lung detail studies
        "LUNG": 1,  # GE, for interstitial lung pathology
        "SOFT": 3,  # GE, for tissues with similar densities
        "STANDARD": 3,  # GE, Standard Body
        "BONE": 4,  # GE, for sharp bone detail
        "YA": 1,  # Philips sharp 'Y-Sharp(YA)'
        "YB": 3,  # Philips sharp 'Y-Detail(YA)'
        "YD": 2,  # Philips Lung Kernel
        "B": 4,  # Philips Standard soft tissue Kernel
        "21": 1,  # Hitachi SCENARIA lung
        "31": 3,  # Hitachi SCENARIA soft tissue
        "UNKNOWN": 9,
    },
)

for resolution_factor in range(29, 75):
    """Siemens kernel"""
    if resolution_factor < 44:  # Standard
        _LOADER_PARAMS.kernel_priority["I{:02d}f".format(resolution_factor)] = 4  # SAFIRE for soft tissue
        _LOADER_PARAMS.kernel_priority["B{:02d}f".format(resolution_factor)] = 3
    elif resolution_factor < 56:  # Medium / Medium Sharp
        _LOADER_PARAMS.kernel_priority["B{:02d}f".format(resolution_factor)] = 2
        _LOADER_PARAMS.kernel_priority["Br{:02d}f".format(resolution_factor)] = 2
    else:  # Sharp
        _LOADER_PARAMS.kernel_priority["B{:02d}f".format(resolution_factor)] = 1
for sharpness in range(11, 87):
    """Toshiba kernel"""
    if sharpness < 49:  # Toshiba Body/Head with BHC
        _LOADER_PARAMS.kernel_priority["FC{:02d}".format(sharpness)] = 6
    elif sharpness < 54:  # Standard Lung
        _LOADER_PARAMS.kernel_priority["FC{:02d}".format(sharpness)] = 3
    elif sharpness < 57:  # Standard Lung Denoised
        _LOADER_PARAMS.kernel_priority["FC{:02d}".format(sharpness)] = 1
    elif sharpness < 82:  # Bone
        _LOADER_PARAMS.kernel_priority["FC{:02d}".format(sharpness)] = 6
    else:  # High res Lung
        _LOADER_PARAMS.kernel_priority["FC{:02d}".format(sharpness)] = 2


def uncompressor(path_to_slice, path_to_save):
    dcm_name = os.path.split(path_to_slice)[-1]
    if os.path.splitext(dcm_name)[1] == "":
        dcm_name += ".dcm"

    path_to_new_slice = os.path.join(path_to_save, dcm_name)
    subprocess.run(shlex.split("gdcmconv --raw '{}' '{}'".format(path_to_slice, path_to_new_slice)))


def load_scan(path, group_separator="0x0020000E", force=2, **kwargs):
    return DicomLoader(_LOADER_PARAMS).load_dicom(path, group_separator=group_separator, force=force, **kwargs)


def load_mhd2numpy(path):
    try:
        itkimage = sitk.ReadImage(path)

    except Exception as e:
        print("load_mhd2numpy exception occured: {}".format(e))
        return None, None

    numpy_image = sitk.GetArrayFromImage(itkimage)
    numpy_spacing = np.array(list(reversed(itkimage.GetSpacing())))  # spacing of voxels in world coordinates (mm)

    return numpy_image, numpy_spacing


class DicomHandler:
    MAX_FILE_SIZE = 1024 * 1024 * 4

    def __init__(self, loader_params: collections.namedtuple = _LOADER_PARAMS):
        self.params = loader_params
        self.dicom_slices = None
        self.dicom_paths = None
        self.set_force()

    def set_force(self, force: int = 2):
        self.force = force

    def _get_spacing_between_slices(self, dicom_paths, exit_code=0):
        try:
            dicom_slices = [dicom.read_file(x, force=True if self.force > 3 else False) for x in dicom_paths]
            dicom_slices = sorted(dicom_slices, key=lambda x: float(x.ImagePositionPatient[2]))
            return np.abs(dicom_slices[1].ImagePositionPatient[2] - dicom_slices[2].ImagePositionPatient[2])

        except:
            return 0.0 if exit_code == 0 else 1.0

    def _get_dicom_list(self, path_to_dir, max_file_size=MAX_FILE_SIZE):
        self.path_to_dir = path_to_dir
        if self.params.read_dcm_only:
            dicom_list = glob("{}/*.dcm".format(path_to_dir))
            dicom_list += glob("{}/*.DCM".format(path_to_dir))

        else:
            files = [os.path.join(directory, f) for directory, _, files in os.walk(path_to_dir) for f in files]

            # Filter files that are equal or larger than max_file_size
            dicom_list = [f for f in files if os.path.getsize(f) < max_file_size]

            filtered_num = len(files) - len(dicom_list)
            if filtered_num:
                warnings.warn(
                    f"There were {filtered_num} files that were filtered out. " f"File size threshold: {max_file_size}"
                )

        if len(dicom_list) < 1:
            raise DicomLoadException("Cannot find DICOM files : {}".format(path_to_dir))

        return dicom_list

    def _get_grouped_list(self, dicom_list, group_separator, verbose=False):
        self.dicom_groups = {}
        self.slice_thickness = {}
        self.kernel_dict = {}
        for path_to_slice in dicom_list:
            try:
                # Ignore dicom files which `SOPClassUID` is not suitable.
                # `force=True` is to load raw file which is not DICOM format.
                _slice = dicom.read_file(path_to_slice, force=True)
                if not "1.2.840.10008.5.1.4.1.1.2" in _slice.SOPClassUID:
                    continue

                # Identify Axial slice using ImageOrientationPatient attribute.
                round_orientation = list(map(round, _slice.ImageOrientationPatient))
                _IOP = np.array(round_orientation, dtype=np.int32)
                _axial_direction = np.array([1, 0, 0, 0, 1, 0])
                if np.count_nonzero(_IOP - _axial_direction) > 0 and np.count_nonzero(_IOP + _axial_direction) > 0:
                    continue

                # Identify whether DICOM file is compressed or not.
                _TransferSyntaxUID = _slice.file_meta.TransferSyntaxUID
                self.is_compressed = (
                    True
                    if "1.2.840.10008.1.2.4" in _TransferSyntaxUID
                    or "1.2.840.10008.1.2.5" in _TransferSyntaxUID
                    or "1.2.840.10008.1.2.6" in _TransferSyntaxUID
                    else False
                )

            except:
                # In case of some attribute is missing or unreadable.
                continue

            try:
                # Cluster dicom files to each group with separator from args.
                group_uid = str(_slice[group_separator].value)
                if len(group_uid) == 0:
                    raise DicomLoadException("Empty separator : {}".format(_slice[group_separator]))

            except Exception as e:
                raise DicomLoadException("Unknown attribute : {}".format(e))

            if group_uid in self.dicom_groups:
                # Validate `SliceThickness` of each slices if exists.
                if self.is_slice_thickness_exists:
                    if self.slice_thickness[group_uid] != _slice.SliceThickness:
                        _msg = "Unexpected SliceThickness : {}".format(path_to_slice)
                        if self.force >= 3:
                            if verbose:
                                warnings.warn(_msg)

                        else:
                            raise DicomLoadException(_msg)

                self.dicom_groups[group_uid] += [path_to_slice]

            else:
                try:
                    # Make `kernel_dict` to determine best case for inference.
                    if type(_slice.ConvolutionKernel) == dicom.multival.MultiValue:
                        convolution_kernel = _slice.ConvolutionKernel[0]

                    else:
                        convolution_kernel = _slice.ConvolutionKernel

                except Exception as e:
                    # Anyway, convolution_kernel is type-3 attribute.
                    if verbose:
                        warnings.warn(str(e))
                    convolution_kernel = "NOT_EXIST"

                if convolution_kernel in self.params.kernel_priority.keys():
                    if self.params.kernel_priority[convolution_kernel] == "NOT_LUNG":
                        continue

                    self.kernel_dict[group_uid] = convolution_kernel

                else:
                    self.kernel_dict[group_uid] = "UNKNOWN"

                self.dicom_groups[group_uid] = [path_to_slice]
                # Because `SliceThickness` is defined as type 2, it means
                # `Required, but Empty if Unknown`, we have to re-define if
                # `SliceThickness` is empty or odd-value. This phase check
                # the meta of EACH dicom slice, and make flag for restoring
                # `SliceThickness`
                try:
                    self.is_slice_thickness_exists = True
                    self.slice_thickness[group_uid] = float(_slice.SliceThickness)

                except:
                    if verbose:
                        warnings.warn("SliceThickness is not found\n   UID :" + group_uid)
                    self.is_slice_thickness_exists = False
                    self.slice_thickness[group_uid] = None

    def _get_target_group_uid(
        self,
        verbose=False,
        min_length: float = 180.0,
        max_length: float = 720.0,
        preferred_recon_algo=None,
    ):
        if len(self.dicom_groups) > 1:
            _msg = "Contains multiple dicom series : {}\n".format(self.path_to_dir)
            if self.force < 1:
                raise DicomLoadException(_msg)

        try:
            _msg = _msg if "_msg" in locals() else ""
            for _group_uid, _group in self.dicom_groups.copy().items():
                if self.slice_thickness[_group_uid] is None:
                    if not self.force < 4:
                        continue
                    _msg = (
                        _msg + " - A DICOM series with no SliceThickness "
                        "is automatically removed by `force < 4` option. " + "\n   UID :" + _group_uid
                    )
                    del self.dicom_groups[_group_uid]
                    del self.kernel_dict[_group_uid]
                    del self.slice_thickness[_group_uid]

                elif len(_group) * self.slice_thickness[_group_uid] < min_length:
                    _spacing_between_slices = self._get_spacing_between_slices(_group)
                    if len(_group) * _spacing_between_slices < min_length:
                        _msg = (
                            f"{_msg} - A DICOM series which is too short to inference is automatically removed. \n"
                            f"   UID: {_group_uid}"
                        )
                        del self.dicom_groups[_group_uid]
                        del self.kernel_dict[_group_uid]
                        del self.slice_thickness[_group_uid]

                    else:
                        _msg = _msg + " - Interslice gap exists.   UID :" + _group_uid

                elif len(_group) * self.slice_thickness[_group_uid] > max_length:
                    _spacing_between_slices = self._get_spacing_between_slices(_group, exit_code=1)
                    if len(_group) * _spacing_between_slices > max_length:
                        _msg = (
                            _msg
                            + " - A DICOM series which is too long "
                            + " to inference is automatically removed. "
                            + "\n   UID :"
                            + _group_uid
                        )
                        del self.dicom_groups[_group_uid]
                        del self.kernel_dict[_group_uid]
                        del self.slice_thickness[_group_uid]

                    else:
                        _msg = _msg + " - Interslice overlapping exists. " + "UID :" + _group_uid

            if preferred_recon_algo:
                try:
                    target_candidate = [
                        uid
                        for uid, algo in list(self.kernel_dict.items())
                        if str(algo).upper() == str(preferred_recon_algo).upper()
                    ]
                    self.target_group_uid = min(target_candidate, key=lambda x: len(self.dicom_groups[x]))
                    return self.target_group_uid

                except:
                    _msg = (
                        _msg + "- A DICOM series reconstructed with the" + f" {preferred_recon_algo} cannot be found."
                    )

            _slice_thickness = [x for x in self.slice_thickness.values()]
            _slice_thickness.sort()
            if len(_slice_thickness) > 1:
                if _slice_thickness[0] != _slice_thickness[1]:
                    self.target_group_uid = min(self.slice_thickness, key=self.slice_thickness.get)
                    _msg = (
                        _msg
                        + "- A DICOM series with the most slices"
                        + " is selected by `force >= 1` option. UID :"
                        + self.target_group_uid
                    )
                    if verbose and len(_msg) > 0:
                        warnings.warn(_msg)
                    return self.target_group_uid

                else:
                    for _group_uid, _group in self.dicom_groups.copy().items():
                        if self.slice_thickness[_group_uid] > _slice_thickness[0]:
                            del self.dicom_groups[_group_uid]
                            del self.kernel_dict[_group_uid]
                            del self.slice_thickness[_group_uid]

            if not len(self.kernel_dict) >= 1:
                raise DicomLoadException("No remaining DICOM series")

            self.target_group_uid = min(
                self.kernel_dict, key=lambda x: self.params.kernel_priority[self.kernel_dict[x]]
            )
            if len(self.kernel_dict) > 1:
                _msg = (
                    _msg
                    + "\n - A DICOM series using Following kernel"
                    + " is selected as inference target: {}".format(self.kernel_dict[self.target_group_uid])
                )

            if verbose and len(_msg) > 0:
                warnings.warn(_msg)
            return self.target_group_uid

        except:
            raise DicomLoadException(_msg + "\n Cannot find compatible DICOM series : {}".format(self.path_to_dir))

    def _get_target_list(self, uncompress=True, order_by_instance_number=False, use_multiprocessing=True):
        if uncompress and self.is_compressed:
            try:
                path_to_uncompressed = tempfile.mkdtemp()
                # Clear target folder to prevent side-effect
                if len(glob("{}/*.dcm".format(path_to_uncompressed))) > 0:
                    subprocess.run("rm {}/*.dcm".format(path_to_uncompressed), shell=True)

            except PermissionError:
                # Basically, You will not get the `PermissionError` at this
                # phase, because `tempfile.mkdtemp()` uses the path of `/tmp/*'
                # have the permission of `777`. This phase is ready for very
                # unique cases, where the `user` of this process cannot access
                # the `/tmp/*` by some unexpected reason. Instead of using the
                # default temporary folder designated by OS,
                # we use `/path/to/loader.py/temp/pid`
                warnings.warn("Cannot access the default temporary folder.")
                path_to_uncompressed = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "temp", str(os.getpid())
                )
                if os.path.exists(os.path.join(path_to_uncompressed)):
                    subprocess.run("rm {}/*.dcm".format(path_to_uncompressed), shell=True)

                else:
                    os.makedirs(path_to_uncompressed)

            target_dicoms = self.dicom_groups[self.target_group_uid]
            if use_multiprocessing:
                pool = Pool(processes=self.params.uncompress_processor)
                try:
                    pool.map(partial(uncompressor, path_to_save=path_to_uncompressed), target_dicoms)
                finally:
                    pool.close()
                    pool.join()
            else:
                for i_slice in target_dicoms:
                    uncompressor(i_slice, path_to_save=path_to_uncompressed)

            try:
                # Align path of uncompressed and original ones.
                path_to_load = glob("{}/*.dcm".format(path_to_uncompressed))
                self.dicom_paths = [
                    os.path.join(os.path.split(y)[0], os.path.split(x)[-1])
                    for x, y in zip(path_to_load, self.dicom_groups[self.target_group_uid])
                ]

            except Exception as e:
                raise DicomLoadException(e)

            self.dicom_slices = [dicom.read_file(x, force=True if self.force > 3 else False) for x in path_to_load]
            if self.params.delete_uncompressed:
                try:
                    shutil.rmtree(path_to_uncompressed)

                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise DicomLoadException(e)

        else:
            self.dicom_paths = self.dicom_groups[self.target_group_uid]
            self.dicom_slices = [dicom.read_file(x, force=True if self.force > 3 else False) for x in self.dicom_paths]

        zipped = list(zip(self.dicom_slices, self.dicom_paths))
        if order_by_instance_number:
            zipped.sort(key=lambda x: int(x[0].InstanceNumber))

        else:
            zipped.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))

        self.dicom_slices = list(list(zip(*zipped))[0])
        self.dicom_paths = list(list(zip(*zipped))[1])
        return self.dicom_slices, self.dicom_paths

    def _check_consistency(self, replace_slice_thickness=True, verbose=True):
        if self.is_slice_thickness_exists:
            slice_thickness = self.slice_thickness[self.target_group_uid]
            spacing_between_slices, sbs_counts = np.unique(
                np.abs(
                    [
                        self.dicom_slices[idx].ImagePositionPatient[2]
                        - self.dicom_slices[idx + 1].ImagePositionPatient[2]
                        for idx in [3, 6, -6, -3]
                    ]
                ),
                return_counts=True,
            )
            spacing_between_slices = spacing_between_slices[np.argmax(sbs_counts)]
            # Check the value of `SpacingBetweenSlices` and `SliceThickness`
            is_SliceThickness_same_with_SpacingBetweenSlices = bool(
                np.abs(spacing_between_slices - slice_thickness) < 0.001
            )
            if not is_SliceThickness_same_with_SpacingBetweenSlices:
                # Put priority on `SpacingBetweenSlices` than `SliceThickness`
                self.original_slice_thickness = slice_thickness
                _msg = (
                    "SliceThickness is not matched with spacing between "
                    + "slices. Spacing between slicess is saved to slices."
                    + "(SliceThickness : {}, Spacing : {})".format(
                        self.original_slice_thickness, spacing_between_slices
                    )
                )
                if verbose:
                    warnings.warn(_msg)

        else:
            try:
                slice_thickness = np.abs(
                    self.dicom_slices[1].ImagePositionPatient[2] - self.dicom_slices[2].ImagePositionPatient[2]
                )
                spacing_between_slices = slice_thickness

            except:
                if verbose:
                    warnings.warn("Cannot find ImagePositionPatient")
                try:
                    slice_thickness = np.abs(self.dicom_slices[1].SliceLocation - self.dicom_slices[2].SliceLocation)
                    spacing_between_slices = slice_thickness

                except:
                    self.dicom_slices = []
                    self.dicom_paths = []
                    raise DicomLoadException("Fail to allocate SliceThickness.")

            is_SliceThickness_same_with_SpacingBetweenSlices = True

        self.NonConstant_SpacingBetweenSlices = 0
        for idx in range(len(self.dicom_slices)):
            self.dicom_slices[idx].SpacingBetweenSlices = spacing_between_slices
            if replace_slice_thickness:
                self.dicom_slices[idx].SliceThickness = spacing_between_slices

            if idx != len(self.dicom_slices) - 1:
                diff = np.abs(
                    np.array(self.dicom_slices[idx].ImagePositionPatient)
                    - np.array(self.dicom_slices[idx + 1].ImagePositionPatient)
                )
                self.NonConstant_SpacingBetweenSlices += bool(
                    np.abs(diff[2] - spacing_between_slices) > 0.01 or diff[0] > 0.001 or diff[1] > 0.001
                )
                if diff[0] > 0.001 or diff[1] > 0.001:
                    _msg = "Unsuitable ImagePositionPatient is detected. " + "(another image included?) : {}".format(
                        idx
                    )
                    if verbose:
                        warnings.warn(_msg)
                    if self.force < 3:
                        self.dicom_slices = []
                        self.dicom_paths = []
                        raise DicomLoadException(_msg)

                if np.abs(diff[2] - spacing_between_slices) > 0.1:
                    _msg = (
                        "Unsuitable SpacingBetweenSlices is detected. "
                        + "(there may be missing or overlapped dicom slices "
                        + "between slice no {} and {})".format(idx, idx + 1)
                    )
                    if self.force < 2:
                        self.dicom_slices = []
                        self.dicom_paths = []
                        raise DicomLoadException(_msg)

        if self.NonConstant_SpacingBetweenSlices > 0:
            if self.NonConstant_SpacingBetweenSlices > self.params.patience_for_inconsistency:
                _msg = (
                    "The number of unsuitable spacing between slices are"
                    + " exceeding patience_for_inconsistency({}).".format(self.params.patience_for_inconsistency)
                    + " The order of slices maybe damaged."
                )
                if self.force < 4:
                    self.dicom_slices = []
                    self.dicom_paths = []
                    raise DicomLoadException(_msg)

            _msg = (
                "SpacingBetweenSlices is not constant : Calculated "
                + f"SpacingBetweenSlices ({spacing_between_slices}) from "
                + "ImagePositionPatient will be adopted for interpolation."
            )
            if verbose:
                warnings.warn(_msg)

        return self.dicom_slices, self.dicom_paths


class DicomLoader(DicomHandler):
    def __init__(self, loader_params: collections.namedtuple = _LOADER_PARAMS):
        super().__init__(loader_params)
        self.params = loader_params

    def load_dicom(
        self,
        path_to_dir,
        uncompress=True,
        check_consistency=True,
        replace_slice_thickness=False,
        order_by_instance_number=False,
        force=2,
        group_separator="0x0020000E",
        preferred_recon_algo=None,
        verbose=True,
        use_multiprocessing=True,
    ):
        print("FORCE IS FORCE IS FORCE IS", force)
        """Universal DICOM loader.
        This function makes well-aligned list of DICOM slices from DICOM series
        following DICOM standards as far as possible. It means that we mainly
        construct the list using attributes of type `required`. And when we
        cannot find the proper information, we can get specific errors about
        that and also restore the information with related attributes.
        Args:
            path_to_dir : String of the path of target DICOM series.
            uncompress : Boolean of whether to rebuild DICOM series with image
                uncompressed or not. Uncompression is executed by 'gdcmconv',
                so as gdcmconv should be accessible at terminal.
            check_consistency : Boolean of whether to check consistency of each
                dicom slice about slice thickness and spacing between slices.
                If False, unless there is a fatal problem with loading, loader
                returns dicom_slices and dicom_paths will be provided. You can
                use this boolean for debugging, and other like things.
            replace_slice_thickness : Boolean of whether to replace each
                SliceThickness (0018, 0050) with Spacing between slices.
                IF False, new attribute `SpacingBetweenSlice` will be created
                to each slices.
            order_by_instance_number : Boolean of whether to sort slices with
                InstanceNumber (0020, 0013).
            group_separator : String of the DICOM tag stated with 0x. Defaults
                to '0x0020000E', SeriesInstanceUID (0020, 000E). This argument
                determines which tags will be used in grouping dicom slices.
                Defaults work best in many cases but could fail when loading
                manipulated, or damaged Series. Alternative attributes like
                SeriesNumber (0020, 0011) could be helpful.
            preferred_recon_algo : String specifying the preffred reconstruction
                algorithm to load. If the loader cannot find the exact algorithm,
                loader will load the dicom series with its default logic.
            force : Integer between 0 and 4. It determines the sensitivity of
                this function when the function encounter the error. Surely it
                would be expanded when we update this function. Note that the
                setting of larger numbers include the smaller numbers's one.
                - if force == 0:
                    Do not permit any kind of exception. `path_to_dir` must
                    include only one case in folder, and there should be no
                    expected error like unconsistency around spacing, etc,.
                - if force == 1:
                    Ignore the multi-series including error. ConvoluitonKernel,
                    SliceThickness, number of slices, are considered to
                    determine `target_group_uid`.
                - if force == 2:
                    Ignore the small amounts of DICOM missing or overlapping.
                    You can adjust the amounts of slices by using argument of
                    loader_params, `patience for missing`. Defaults to 3.
                - if force == 3:
                    TBD.
                - if force == 4:
                    TRY TO LOAD DICOM ANYWAY!
        Returns:
            dicom_slices : list of the dicom slices.
            dicom_paths : list of the path of each slices.
        """
        self.path_to_dir = path_to_dir
        self.set_force(force)
        self._get_grouped_list(self._get_dicom_list(path_to_dir), group_separator, verbose)
        if self.force <= 2:
            min_length = 180.0
            max_length = 720.0
        else:
            min_length = 96.0  # to prevent data pipeline error.
            max_length = float("inf")
        self._get_target_group_uid(
            verbose,
            min_length=min_length,
            max_length=max_length,
            preferred_recon_algo=preferred_recon_algo,
        )
        self.dicom_slices, self.dicom_paths = self._get_target_list(
            uncompress, order_by_instance_number, use_multiprocessing
        )
        if check_consistency:
            self.dicom_slices, self.dicom_paths = self._check_consistency(replace_slice_thickness, verbose)

        return self.dicom_slices, self.dicom_paths

    def get_dicom_slices(self, path_to_dir=None):
        if not self.dicom_slices and path_to_dir != self.path_to_dir:
            _, _ = self.load_dicom(path_to_dir)
        return self.dicom_slices

    def get_dicom_paths(self, path_to_dir=None):
        if not self.dicom_slices and path_to_dir != self.path_to_dir:
            _, _ = self.load_dicom(path_to_dir)
        return self.dicom_paths
