import cv2
from PIL import Image
import numpy as np
import os


def augmented(image,Extension, dor,image_path,category,flip_i,rotation_flip_j,rotation_k,filename):
    image_flip = cv2.flip(image, dor)
    cv2.imwrite(image_path + "/" + category +"."+ str(flip_i) +"."+ Extension, image_flip)
    input_image = np.array(Image.open(image_path + "/" + category +"."+ str(flip_i) +"."+ Extension))
    Image.fromarray(np.rot90(input_image,3)).save(image_path + "/" + category +"."+ str(rotation_flip_j) +"."+ Extension)
    input_image = np.array(Image.open(image_path+"/"+filename))
    Image.fromarray(np.rot90(input_image)).save(image_path + "/" + category +"."+ str(rotation_k) +"."+ Extension)

def test(image_path):
    filenames = os.listdir(image_path)
    for filename in filenames:

        # Jumlah Foto test
        jumlah_foto = 150
        category = filename.split('.')[0]
        i = filename.split('.')[1]
        Extension = filename.split('.')[2]

        # index Image
        flip_i = int(i) + jumlah_foto
        rotation_flip_j = flip_i + jumlah_foto
        rotation_k = rotation_flip_j + jumlah_foto

        # Path
        image = cv2.imread(image_path + "/" + filename)

        if category == 'Autistic':

            # Untuk ngetes
            # print(image_path+"/"+filename)
            # print(i + " jadi " + str(flip_i) + " jadi " + str(rotation_flip_j) + " Jadi " + str(rotation_k) +"."+ Extension)

            augmented(image, Extension, 1, image_path, category, flip_i, rotation_flip_j, rotation_k, filename)
        else:
            augmented(image, Extension, 1, image_path, category, flip_i, rotation_flip_j, rotation_k, filename)

def train(image_path):
    filenames = os.listdir(image_path)
    for filename in filenames:

        # Jumlah Foto train
        jumlah_foto = 1270
        category = filename.split('.')[0]
        i = filename.split('.')[1]
        Extension = filename.split('.')[2]

        # index Image
        flip_i = int(i) + jumlah_foto
        rotation_flip_j = flip_i + jumlah_foto
        rotation_k = rotation_flip_j + jumlah_foto

        # Path
        image = cv2.imread(image_path + "/" + filename)

        if category == 'Autistic':
            augmented(image, Extension, 1, image_path, category, flip_i, rotation_flip_j, rotation_k, filename)
        else:
            augmented(image, Extension, 1, image_path, category, flip_i, rotation_flip_j, rotation_k, filename)

def valid_non_autis(image_path_valid_non_autistic):
    filenames = os.listdir(image_path_valid_autistic)
    for filename in filenames:
        jumlah_foto = 50

        name = filename.split('.')[0]
        Extension = filename.split('.')[1]
        # print(filename + " and " + name + " and " + Extension)

        flip_i = int(name) + jumlah_foto
        rotation_flip_j = flip_i + jumlah_foto
        rotation_k = rotation_flip_j + jumlah_foto

        # print(type(flip_i))
        image = cv2.imread(image_path_valid_autistic + "/" + filename)

        image_flip = cv2.flip(image, 1)
        cv2.imwrite(image_path_valid_autistic + "/" + str(flip_i) + "." + Extension, image_flip)
        input_image = np.array(Image.open(image_path_valid_autistic + "/" + str(flip_i) + "." + Extension))
        Image.fromarray(np.rot90(input_image, 3)).save(image_path_valid_autistic + str(rotation_flip_j) + "." + Extension)
        input_image = np.array(Image.open(image_path_valid_autistic + "/" + filename))
        Image.fromarray(np.rot90(input_image)).save(image_path_valid_autistic + "/" + str(rotation_k) + "." + Extension)

def valid_autis(image_path_valid_autistic):
    filenames = os.listdir(image_path_valid_autistic)
    for filename in filenames:
        jumlah_foto = 50

        name = filename.split('.')[0]
        Extension = filename.split('.')[1]
        # print(filename + " and " + name + " and " + Extension)

        flip_i = int(name) + jumlah_foto
        rotation_flip_j = flip_i + jumlah_foto
        rotation_k = rotation_flip_j + jumlah_foto

        # print(type(flip_i))
        image = cv2.imread(image_path_valid_autistic + "/" + filename)

        image_flip = cv2.flip(image, 1)
        cv2.imwrite(image_path_valid_autistic + "/" + str(flip_i) + "." + Extension, image_flip)
        input_image = np.array(Image.open(image_path_valid_autistic + "/" + str(flip_i) + "." + Extension))
        Image.fromarray(np.rot90(input_image, 3)).save(image_path_valid_autistic + str(rotation_flip_j) + "." + Extension)
        input_image = np.array(Image.open(image_path_valid_autistic + "/" + filename))
        Image.fromarray(np.rot90(input_image)).save(image_path_valid_autistic + "/" + str(rotation_k) + "." + Extension)


if __name__ == '__main__':
    # path = "tes/before/0001.jpg"
    # image = cv2.imread(path)
    # flip_image(image, 1,path)  # vertical

    image_path_test  = "AutismDataset/test"
    image_path_train = "AutismDataset/train"
    image_path_valid_autistic = "AutismDataset/valid/Autistic"
    image_path_valid_non_autistic = "AutismDataset/valid/Non_Autistic"

    # uncoment this function
    # test(image_path_test)
    # train(image_path_train)

    # uncoment this function
    # valid_autis(image_path_valid_autistic)
    valid_autis(image_path_valid_non_autistic)

