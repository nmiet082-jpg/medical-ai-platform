import numpy as np
import SimpleITK as sitk


# The trained model expects 1 mm isotropic spacing
TARGET_SPACING = (1.0, 1.0, 1.0)

# Model preprocessing
HU_MIN = -1000
HU_MAX = 400

# Model input size
PATCH_SIZE = (32, 32, 32)


def load_dicom_series(dicom_folder):
    """
    Load a CT DICOM series from a folder.

    Returns:
        image: SimpleITK 3D image
    """

    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)

    if not series_ids:
        raise ValueError(
            f"No DICOM series found in: {dicom_folder}"
        )

    # Use the first series found
    series_file_names = reader.GetGDCMSeriesFileNames(
        dicom_folder,
        series_ids[0]
    )

    reader.SetFileNames(series_file_names)

    image = reader.Execute()

    return image


def resample_to_1mm(image):
    """
    Resample CT volume to 1 mm isotropic spacing.
    """

    original_spacing = np.array(image.GetSpacing(), dtype=float)
    original_size = np.array(image.GetSize(), dtype=int)

    target_spacing = np.array(
        TARGET_SPACING,
        dtype=float
    )

    new_size = np.round(
        original_size * original_spacing / target_spacing
    ).astype(int)

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(
        tuple(target_spacing)
    )

    resampler.SetSize(
        [int(x) for x in new_size]
    )

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetTransform(
        sitk.Transform()
    )

    # CT intensity interpolation
    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    resampled = resampler.Execute(image)

    return resampled


def normalize_ct(image):
    """
    Convert CT image to numpy and apply:
        HU clipping [-1000, 400]
        normalization [0, 1]
    """

    volume = sitk.GetArrayFromImage(
        image
    ).astype(np.float32)

    # Clip Hounsfield Units
    volume = np.clip(
        volume,
        HU_MIN,
        HU_MAX
    )

    # Normalize to [0, 1]
    volume = (
        volume - HU_MIN
    ) / (
        HU_MAX - HU_MIN
    )

    return volume


def extract_center_patch(volume):
    """
    Extract a 32x32x32 patch from the center
    of the volume.

    IMPORTANT:
    This is only a technical preprocessing test.
    It does NOT locate a lung nodule.
    """

    depth, height, width = volume.shape

    center_z = depth // 2
    center_y = height // 2
    center_x = width // 2

    half = 16

    z1 = center_z - half
    z2 = center_z + half

    y1 = center_y - half
    y2 = center_y + half

    x1 = center_x - half
    x2 = center_x + half

    # Check that the volume is large enough
    if (
        z1 < 0 or
        y1 < 0 or
        x1 < 0 or
        z2 > depth or
        y2 > height or
        x2 > width
    ):
        raise ValueError(
            "CT volume is too small to extract "
            "a 32x32x32 center patch."
        )

    patch = volume[
        z1:z2,
        y1:y2,
        x1:x2
    ]

    return patch


def prepare_ct_patch(dicom_folder):
    """
    Complete CT preprocessing pipeline.

    DICOM series
        -> 1 mm resampling
        -> HU clipping
        -> normalization
        -> 32x32x32 patch
    """

    print("Loading DICOM series...")

    image = load_dicom_series(
        dicom_folder
    )

    print(
        "Original size:",
        image.GetSize()
    )

    print(
        "Original spacing:",
        image.GetSpacing()
    )

    print("Resampling to 1 mm isotropic spacing...")

    image = resample_to_1mm(image)

    print(
        "Resampled size:",
        image.GetSize()
    )

    print(
        "Resampled spacing:",
        image.GetSpacing()
    )

    print("Applying HU preprocessing...")

    volume = normalize_ct(image)

    print(
        "Volume shape:",
        volume.shape
    )

    print("Extracting 32x32x32 test patch...")

    patch = extract_center_patch(
        volume
    )

    print(
        "Patch shape:",
        patch.shape
    )

    return patch