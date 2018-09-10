import xml.etree.ElementTree as ET
import os

folder = 'data_source/data/train_annot_folder'
FROM = "/Users/epinyoanun/Desktop/EXIT/tutorial/machine_learning/jupyter/experiment/keras-yolo3-master/train_image_folder/"
TO = "/Users/epinyoanun/Desktop/EXIT/tutorial/machine_learning/jupyter/experiment/keras-yolo3-master/data_source/data/train_image_folder/"

for filename in os.listdir(folder):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(folder, filename)
    tree = ET.parse(fullname)
    root = tree.getroot()
    for path in root.findall('./path'):
        path.text = path.text.replace(FROM, TO)
        # print('path.text:', path.text)
    tree.write(fullname)