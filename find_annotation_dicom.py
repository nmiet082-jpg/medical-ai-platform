import os
import pydicom
import xml.etree.ElementTree as ET

dicom_folder = r"C:\Users\gawad\Downloads\lung_nodule_model\lidc_idri\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\CT_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"

xml_file = r"C:\Users\gawad\Downloads\LIDC-XML-only\tcia-lidc-xml\185\069.xml"

# Get all DICOM SOP UIDs
dicom_uids = {}

for filename in os.listdir(dicom_folder):
    if filename.lower().endswith(".dcm"):
        filepath = os.path.join(dicom_folder, filename)
        dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
        dicom_uids[dcm.SOPInstanceUID] = filepath

print("DICOM slices found:", len(dicom_uids))

# Read XML
root = ET.parse(xml_file).getroot()

matches = []

for roi in root.iter():
    if not roi.tag.endswith("roi"):
        continue

    sop = None
    z = None
    xs = []
    ys = []

    for element in roi.iter():
        if element.tag.endswith("imageSOP_UID"):
            sop = element.text
        elif element.tag.endswith("imageZposition"):
            z = element.text
        elif element.tag.endswith("xCoord"):
            xs.append(float(element.text))
        elif element.tag.endswith("yCoord"):
            ys.append(float(element.text))

    if sop in dicom_uids:
        matches.append((sop, z, xs, ys, dicom_uids[sop]))

print("Matching annotated slices:", len(matches))

for i, (sop, z, xs, ys, filepath) in enumerate(matches[:5]):
    print()
    print("MATCH", i + 1)
    print("SOP:", sop)
    print("Z:", z)
    print("X center:", round(sum(xs) / len(xs), 1))
    print("Y center:", round(sum(ys) / len(ys), 1))
    print("Points:", len(xs))
    print("DICOM:", filepath)