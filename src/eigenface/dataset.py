from os import listdir
from os.path import join, isfile, isdir, splitext
from PIL import Image
from eigenface.vector import Vector
from eigenface.matrix import Matrix


SUPPORTED_FILES = {".pgm", ".png", ".jpg", ".jpeg", ".bmp"}


def load_dataset(folder_path: str, height: int, width: int):
    matrix: Matrix = []
    for folder in listdir(folder_path):
        filepath = join(folder_path, folder)
        if not isdir(filepath):
            continue
        for file in listdir(filepath):
            image_path = join(filepath, file)
            _, file_type = splitext(file)
            if (
                isfile(image_path)
                and file_type.lower() in SUPPORTED_FILES
            ):
                vector = load_image_as_vector(image_path, height, width)
                matrix.append(vector)
    if not matrix:
        raise ValueError("no images found in folder")
    return matrix


def load_image_as_vector(image_path: str, height: int, width: int):
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize((width, height))
        pixels: Vector = [p/255 for p in img.getdata()]

    return pixels
