from PIL import Image
from vector import Vector
from matrix import Matrix
from os import listdir
from os.path import join, isfile


def load_dataset(folder_path:str, height:int, width:int):
    matrix: Matrix=[]
    for folder in listdir(folder_path):
        filepath= join(folder_path,folder)
        for file in listdir(filepath):
            image_path=join(filepath,file)
            if isfile(image_path) and len(matrix)< 2510:
                vector=load_image_as_vector(image_path,height,width)
                matrix.append(vector)
    if not matrix:
        raise ValueError("no images found in folder")
    n=len(matrix)
    holdout_size= max(1,round(0.2*n))
    split=n-holdout_size
    test_matrix= matrix[:split]
    training_matrix=matrix[split:]
    return training_matrix, test_matrix


def load_image_as_vector(image_path:str,height:int,width:int):
    with Image.open(image_path) as img:
        img=img.convert("L")
        img = img.resize((width, height))
        pixels:Vector=[p/255 for p in img.getdata()]

    return pixels
