import xml.etree.ElementTree as ET

XML_FILE = r"C:\Users\gawad\Downloads\LIDC-XML-only\tcia-lidc-xml\185\069.xml"

TARGET_SOP = "1.3.6.1.4.1.14519.5.2.1.6279.6001.297813206491522913194774892711"

root = ET.parse(XML_FILE).getroot()

for nodule in root.iter():

    if not nodule.tag.endswith("unblindedReadNodule"):
        continue

    found = False

    for roi in nodule.iter():

        if not roi.tag.endswith("roi"):
            continue

        for element in roi.iter():

            if (
                element.tag.endswith("imageSOP_UID")
                and element.text == TARGET_SOP
            ):
                found = True
                break

        if found:
            break

    if found:

        print("================================")
        print("TARGET NODULE CHARACTERISTICS")
        print("================================")

        wanted = {
            "noduleID",
            "characteristics",
            "subtlety",
            "internalStructure",
            "calcification",
            "sphericity",
            "margin",
            "lobulation",
            "spiculation",
            "texture",
            "malignancy",
        }

        for element in nodule.iter():

            tag = element.tag.split("}")[-1]

            if tag in wanted and element.text:
                print(f"{tag}: {element.text.strip()}")

        break