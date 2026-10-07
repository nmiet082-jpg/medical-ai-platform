import os
import numpy as np
import pydicom
import SimpleITK as sitk
import xml.etree.ElementTree as ET


# ============================================================
# PATHS
# ============================================================

DICOM_FOLDER = r"C:\Users\gawad\Downloads\lung_nodule_model\lidc_idri\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\CT_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"

XML_FILE = r"C:\Users\gawad\Downloads\LIDC-XML-only\tcia-lidc-xml\185\069.xml"

OUTPUT_FILE = r"C:\medical-ai-platform\annotated_nodule_patch.npy"


# ============================================================
# FIND DICOM FILES
# ============================================================

dicom_info = {}

for filename in os.listdir(DICOM_FOLDER):

    if not filename.lower().endswith(".dcm"):
        continue

    filepath = os.path.join(DICOM_FOLDER, filename)

    dcm = pydicom.dcmread(
        filepath,
        stop_before_pixels=True
    )

    dicom_info[dcm.SOPInstanceUID] = {
        "file": filepath,
        "position": [
            float(x)
            for x in dcm.ImagePositionPatient
        ],
        "orientation": [
            float(x)
            for x in dcm.ImageOrientationPatient
        ],
        "spacing": [
            float(x)
            for x in dcm.PixelSpacing
        ],
    }


print("DICOM slices:", len(dicom_info))


# ============================================================
# READ LIDC XML
# ============================================================

root = ET.parse(XML_FILE).getroot()

matches = []

for roi in root.iter():

    if not roi.tag.endswith("roi"):
        continue

    sop = None
    xs = []
    ys = []

    for element in roi.iter():

        if element.tag.endswith("imageSOP_UID"):
            sop = element.text

        elif element.tag.endswith("xCoord"):
            xs.append(float(element.text))

        elif element.tag.endswith("yCoord"):
            ys.append(float(element.text))

    # Only keep ROIs whose DICOM slice exists
    if sop in dicom_info and xs and ys:

        matches.append({
            "sop": sop,
            "xs": xs,
            "ys": ys,
        })


print("Matching annotated ROIs:", len(matches))


# ============================================================
# SELECT THE SAME REAL ANNOTATED ROI
# ============================================================

# We previously selected the 5th matching ROI.
roi = matches[4]

sop = roi["sop"]

xs = roi["xs"]
ys = roi["ys"]

info = dicom_info[sop]


# Calculate center of annotation contour
x_pixel = sum(xs) / len(xs)
y_pixel = sum(ys) / len(ys)


print()
print("Selected annotation:")
print("SOP:", sop)
print("X pixel center:", round(x_pixel, 2))
print("Y pixel center:", round(y_pixel, 2))
print("Contour points:", len(xs))


# ============================================================
# CONVERT DICOM PIXEL COORDINATES TO PHYSICAL COORDINATES
# ============================================================

ipp = np.array(info["position"])

orientation = np.array(info["orientation"])

spacing = np.array(info["spacing"])


# DICOM ImageOrientationPatient:
#
# first 3 values  = row direction
# second 3 values = column direction

row_direction = orientation[:3]

column_direction = orientation[3:]


# IMPORTANT:
# xCoord is the column coordinate.
# yCoord is the row coordinate.
#
# Therefore:
# xCoord uses column direction + column spacing
# yCoord uses row direction + row spacing

physical_point = (
    ipp
    + row_direction * x_pixel * spacing[1]
    + column_direction * y_pixel * spacing[0]
)

physical_point = tuple(
    float(x)
    for x in physical_point
)


print("Physical point:", physical_point)


# ============================================================
# LOAD COMPLETE CT SERIES
# ============================================================

reader = sitk.ImageSeriesReader()

series_files = reader.GetGDCMSeriesFileNames(
    DICOM_FOLDER
)

reader.SetFileNames(series_files)

image = reader.Execute()


print()
print("Original CT:")
print("Size:", image.GetSize())
print("Spacing:", image.GetSpacing())


# ============================================================
# RESAMPLE CT TO 1 MM ISOTROPIC
# ============================================================

original_spacing = image.GetSpacing()

original_size = image.GetSize()

new_spacing = (
    1.0,
    1.0,
    1.0
)


new_size = [
    int(
        round(
            original_size[i]
            * original_spacing[i]
            / new_spacing[i]
        )
    )
    for i in range(3)
]


resampler = sitk.ResampleImageFilter()

resampler.SetOutputSpacing(new_spacing)

resampler.SetSize(new_size)

resampler.SetOutputDirection(
    image.GetDirection()
)

resampler.SetOutputOrigin(
    image.GetOrigin()
)

resampler.SetTransform(
    sitk.Transform()
)

resampler.SetInterpolator(
    sitk.sitkLinear
)


resampled = resampler.Execute(image)


print()
print("Resampled CT:")
print("Size:", resampled.GetSize())
print("Spacing:", resampled.GetSpacing())


# ============================================================
# FIND ANNOTATION LOCATION IN RESAMPLED CT
# ============================================================

center_index = (
    resampled.TransformPhysicalPointToIndex(
        physical_point
    )
)


print()
print(
    "Annotation index in resampled CT:",
    center_index
)


# ============================================================
# CHECK PATCH IS INSIDE THE CT
# ============================================================

patch_size = 32

half = patch_size // 2

cx, cy, cz = center_index

start_x = cx - half
start_y = cy - half
start_z = cz - half

end_x = start_x + patch_size
end_y = start_y + patch_size
end_z = start_z + patch_size

volume_size = resampled.GetSize()

print()
print("Patch bounds:")
print(
    "X:",
    start_x,
    "to",
    end_x
)

print(
    "Y:",
    start_y,
    "to",
    end_y
)

print(
    "Z:",
    start_z,
    "to",
    end_z
)

print("CT size:", volume_size)


if (
    start_x < 0
    or start_y < 0
    or start_z < 0
    or end_x > volume_size[0]
    or end_y > volume_size[1]
    or end_z > volume_size[2]
):

    raise RuntimeError(
        "32x32x32 patch would extend outside the CT volume."
    )


# ============================================================
# EXTRACT 32 x 32 x 32 PATCH
# ============================================================

patch = sitk.RegionOfInterest(
    resampled,
    size=[
        patch_size,
        patch_size,
        patch_size
    ],
    index=[
        start_x,
        start_y,
        start_z
    ]
)


patch_array = sitk.GetArrayFromImage(
    patch
).astype(np.float32)


print()
print("PATCH:")
print("Shape:", patch_array.shape)

print(
    "Minimum HU:",
    float(patch_array.min())
)

print(
    "Maximum HU:",
    float(patch_array.max())
)


# ============================================================
# MODEL PREPROCESSING
# ============================================================

# Clip CT Hounsfield Units
patch_array = np.clip(
    patch_array,
    -1000,
    400
)


# Normalize to [0, 1]
patch_array = (
    patch_array + 1000
) / 1400


print()
print("PREPROCESSED PATCH:")

print(
    "Shape:",
    patch_array.shape
)

print(
    "Minimum:",
    float(patch_array.min())
)

print(
    "Maximum:",
    float(patch_array.max())
)


# ============================================================
# SAVE PATCH
# ============================================================

np.save(
    OUTPUT_FILE,
    patch_array
)


print()
print("Saved patch:")

print(OUTPUT_FILE)