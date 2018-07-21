from PIL import Image
import numpy as np
import os
import os.path


def load_image_to_nparray(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def load_images_in_folder_to_nparray(path):
    img_nparray = None
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        current_image = load_image_to_nparray(os.path.join(path, f))
        if img_nparray is None:
            img_nparray = np.array([current_image])
        else:
            img_nparray = np.concatenate(
                (img_nparray, np.array([current_image])), axis=0)
    return img_nparray


""" EXAMPLE
load_images_in_folder_to_nparray("./data/circles/") """
