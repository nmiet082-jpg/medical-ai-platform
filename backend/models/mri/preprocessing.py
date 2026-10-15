"""Input validation and preprocessing for the supplied research models."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODALITIES = ["flair", "t1", "t1ce", "t2"]

SEGMENTATION_LABELS = {
    0: "background",
    1: "BraTS label 1 (NCR/NET)",
    2: "BraTS label 2 (edema)",
    3: "BraTS label 4 (enhancing tumor)",
}


def _robust_uint8(array):
    """Convert finite image values to uint8 using robust percentile scaling."""
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)

    if not finite.any():
        raise ValueError("Image contains no finite pixel values.")

    arr = np.where(finite, arr, 0.0)
    values = arr[finite]
    lo, hi = np.percentile(values, [1, 99])

    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def load_2d_image(uploaded_file, filename):
    """Read PNG/JPG/JPEG or a single-frame DICOM image for classification."""
    suffix = Path(filename).suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg"}:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image

    if suffix == ".dcm":
        try:
            import pydicom
        except ImportError as exc:
            raise RuntimeError(
                "Install pydicom to read DICOM files."
            ) from exc

        ds = pydicom.dcmread(uploaded_file, force=False)

        if not hasattr(ds, "PixelData"):
            raise ValueError("DICOM file does not contain pixel data.")

        pixels = ds.pixel_array

        if pixels.ndim != 2:
            raise ValueError(
                "Only single-frame 2D DICOM images are supported "
                "by the classifier."
            )

        pixels = pixels.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept

        if str(
            getattr(ds, "PhotometricInterpretation", "")
        ).upper() == "MONOCHROME1":
            pixels = pixels.max() - pixels

        gray = Image.fromarray(_robust_uint8(pixels), mode="L")
        return gray.convert("RGB")

    raise ValueError(
        "Supported classifier formats: DICOM (.dcm), PNG, JPG, JPEG."
    )


def classifier_tensor(pil_image, image_size=224):
    """Resize and normalize an image for the ResNet MRI classifier."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transform(pil_image).unsqueeze(0)


def _load_nifti_from_upload(uploaded_file, nib):
    """Load a NIfTI from a path or a seekable uploaded file object."""
    if isinstance(uploaded_file, (str, Path)):
        return nib.load(str(uploaded_file))

    if not hasattr(uploaded_file, "read"):
        raise TypeError(
            "Each MRI input must be a filesystem path or a "
            "readable uploaded file."
        )

    # Restore the file pointer so repeated reads work where supported.
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    image = nib.load(uploaded_file)

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    return image


def _validate_nifti_geometry(image, filename):
    """Validate basic NIfTI dimensions and physical geometry."""
    if len(image.shape) != 3:
        raise ValueError(
            f"{filename} must be a 3D NIfTI volume; got {image.shape}."
        )

    affine = np.asarray(image.affine, dtype=np.float64)

    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(
            f"{filename} has an invalid or non-finite affine matrix."
        )

    if abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError(
            f"{filename} has singular or invalid spatial geometry."
        )

    return affine


def _center_crop_or_pad(volume, target_shape):
    """Center-crop or zero-pad a 3D array to the requested shape."""
    output = np.zeros(target_shape, dtype=np.float32)
    source_slices = []
    destination_slices = []

    for old_size, new_size in zip(volume.shape, target_shape):
        take = min(old_size, new_size)
        source_start = max((old_size - take) // 2, 0)
        destination_start = max((new_size - take) // 2, 0)

        source_slices.append(
            slice(source_start, source_start + take)
        )
        destination_slices.append(
            slice(destination_start, destination_start + take)
        )

    output[tuple(destination_slices)] = volume[tuple(source_slices)]
    return output


def _normalize_inference_volume(volume, filename):
    """Robustly scale a single modality to [0, 1].

    This is the current compatibility normalization. It is not yet proven
    equivalent to the crop-level normalization used during model training.
    """
    data = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(data)

    if not finite.any():
        raise ValueError(f"{filename} contains no finite voxel values.")

    data = np.where(finite, data, 0.0)
    nonzero = data != 0

    if not nonzero.any():
        return np.zeros_like(data, dtype=np.float32)

    values = data[nonzero]
    low, high = np.percentile(values, [1, 99])

    if high <= low:
        return np.zeros_like(data, dtype=np.float32)

    data = np.clip((data - low) / (high - low), 0.0, 1.0)
    data[~nonzero] = 0.0

    return data.astype(np.float32, copy=False)


def read_nifti_modalities(uploaded_files, crop_size=(32, 96, 96)):
    """Read four spatially aligned NIfTI modalities.

    Expected input order: FLAIR, T1, T1ce, T2.

    Validates:
      - exactly four inputs;
      - 3D volumes;
      - matching array shapes;
      - matching orientation codes;
      - matching affine matrices.

    Returns:
      torch.float32 tensor with shape (4, D, H, W).

    NOTE:
      This function retains center crop/pad for compatibility with the
      existing application. It does not reconstruct a full-volume mask.
      Do not interpret its output as validated full-volume segmentation.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "Install nibabel to read NIfTI segmentation inputs."
        ) from exc

    if len(uploaded_files) != 4:
        raise ValueError(
            "3D segmentation requires exactly four NIfTI files "
            "in this order: FLAIR, T1, T1ce, T2."
        )

    loaded = []
    reference_shape = None
    reference_affine = None
    reference_orientation = None
    reference_name = None

    # Validate all four volumes before preprocessing any of them.
    for uploaded_file in uploaded_files:
        filename = (
            Path(uploaded_file).name
            if isinstance(uploaded_file, (str, Path))
            else getattr(uploaded_file, "name", "uploaded NIfTI")
        )

        image = _load_nifti_from_upload(uploaded_file, nib)
        affine = _validate_nifti_geometry(image, filename)
        orientation = nib.aff2axcodes(affine)

        if reference_shape is None:
            reference_shape = image.shape
            reference_affine = affine.copy()
            reference_orientation = orientation
            reference_name = filename
        else:
            if image.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch: {filename} has shape {image.shape}, "
                    f"but {reference_name} has shape {reference_shape}."
                )

            if orientation != reference_orientation:
                raise ValueError(
                    f"Orientation mismatch: {filename} is {orientation}, "
                    f"but {reference_name} is {reference_orientation}. "
                    "Reorient/resample the volumes to a common reference."
                )

            if not np.allclose(
                affine,
                reference_affine,
                rtol=0.0,
                atol=1e-3,
            ):
                raise ValueError(
                    f"Spatial affine mismatch: {filename} does not align "
                    f"with {reference_name}. Register or resample the "
                    "modalities to a common reference before inference."
                )

        data = np.asarray(image.get_fdata(), dtype=np.float32)

        if not np.isfinite(data).any():
            raise ValueError(
                f"{filename} contains no finite voxel values."
            )

        loaded.append((filename, data))

    if crop_size is None:
        target_shape = None
    else:
        target_shape = tuple(int(v) for v in crop_size)

        if len(target_shape) != 3 or any(v <= 0 for v in target_shape):
            raise ValueError(
                "crop_size must contain three positive integers."
            )

    processed = []

    for filename, data in loaded:
        # Normalize in the NIfTI array domain, then convert x,y,z -> z,y,x.
        data = _normalize_inference_volume(data, filename)
        data = np.transpose(data, (2, 1, 0))

        if target_shape is not None:
            data = _center_crop_or_pad(data, target_shape)

        processed.append(data)

    # All four modalities must remain spatially aligned after preprocessing.
    if any(data.shape != processed[0].shape for data in processed[1:]):
        raise ValueError(
            "The four MRI modalities have different shapes after preprocessing."
        )

    image_array = np.stack(processed, axis=0).astype(
        np.float32, copy=False
    )

    import torch
    return torch.from_numpy(image_array.copy()).float()