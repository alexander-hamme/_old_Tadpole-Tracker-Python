import glob
import os
import cv2

'''
Put this script in the folder with text files you want to convert,
or change `current_dir` to a directory with those files.

Change `save_dir` to the location you want it to save the new xml files.

`image_dir` is the directory containing the images described by the yolo annotations.
'''

current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = current_dir + "/voc_format/"
image_dir = "/home/alex/Documents/sproj/tadpole_dataset/images/"

for filepath in glob.iglob(os.path.join(current_dir, "*.txt")):

    text_values = {
        "folder": "",
        "filename": "",
        "path": "",
        "width": None,
        "height": None,
        "depth": None,
        "class": "",
        "xmin": None,
        "ymin": None,
        "xmax": None,
        "ymax": None
    }

    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)

    print("reading {}.jpg".format(image_dir + base))
    h, w, d = cv2.imread(image_dir + base + ".jpg").shape

    text_values["folder"] = "images"
    text_values["filename"] = base + ".jpg"
    text_values["path"] = image_dir + base + ".jpg"
    text_values["width"] = w
    text_values["height"] = h
    text_values["depth"] = d
    text_values["class"] = "tadpole"

    with open(filename, "r") as in_doc:
        with open(save_dir + base + ".xml", "w") as out_doc:
            data = [l.strip() for l in in_doc.readlines()]

            yolo_numbs = [map(float, strng.split()[1:]) for strng in data]
            # for example [[0.462962962963, 0.488235294118, 0.62962962963, 0.835294117647]]

            """
            Yolo
            <class> <centerX/imageWidth> <centerY/imageHeight> <bboxWidth/imageWidth> <bboxHeight/imageHeight> (We'll index this from 0-4)
            Voc:
            <class> <bboxXMin> <bboxYMin> <bboxXMax> <bboxYMax> (Index this 0-4)

            centerX = yolo[1] * imageWidth
            centerY = yolo[2] * imageHeight
            bboxWidth = yolo[3] * imageWidth
            bboxHeight = yolo[4] * imageHeight

            voc[1] = centerX - (bboxWidth/2)
            voc[2] = centerY - (bboxHeight/2)
            voc[3] = centerX + (bboxWidth/2)
            voc[4] = centerY + (bboxHeight/2)
            """

            bounding_boxes = []

            for lst in yolo_numbs:

                cntr_x = lst[0] * w
                cntr_y = lst[1] * h
                bbox_w = lst[2] * w
                bbox_h = lst[3] * h

                voc_coords = map(int, [
                    cntr_x - (bbox_w / 2) + 0.5,
                    cntr_y - (bbox_h / 2) + 0.5,
                    cntr_x + (bbox_w / 2) + 0.5,
                    cntr_y + (bbox_h / 2 + 0.5)
                ])

                bounding_boxes.append(voc_coords)

            boxes_text = ""
            for box in bounding_boxes:
                boxes_text += """
    <object>
        <name>{0}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{1}</xmin>
            <ymin>{2}</ymin>
            <xmax>{3}</xmax>
            <ymax>{4}</ymax>
        </bndbox>
    </object>""".format(
        text_values.get("class"), *box
    )

            save_text = """<annotation verified="yes">
    <folder>{0}</folder>
    <filename>{1}</filename>
    <path>{2}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{3}</width>
        <height>{4}</height>
        <depth>{5}</depth>
    </size>
    <segmented>0</segmented>{6}
</annotation>""".format(
    text_values.get("folder"),
    text_values.get("filename"),
    text_values.get("path"),
    text_values.get("width"),
    text_values.get("height"),
    text_values.get("depth"),
    boxes_text
)
            out_doc.write(save_text)
