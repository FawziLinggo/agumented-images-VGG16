import cv2
from PIL import Image
import numpy as np
import os

Folder_name = "tes/before/"


def augmented(image,Extension, dor,image_path,category,flip_i,rotation_flip_j,rotation_k,filename):
    image_flip = cv2.flip(image, dor)
    cv2.imwrite(Folder_name + category +"."+ str(flip_i) +"."+ Extension, image_flip)
    input_image = np.array(Image.open(Folder_name + category +"."+ str(flip_i) +"."+ Extension))
    Image.fromarray(np.rot90(input_image,3)).save(Folder_name + category +"."+ str(rotation_flip_j) +"."+ Extension)
    input_image = np.array(Image.open(image_path+"/"+filename))
    Image.fromarray(np.rot90(input_image)).save(Folder_name + category +"."+ str(rotation_k) +"."+ Extension)


# i = Flip (imwrite)
# j = rotation-flip (Image.fromarray)
# k = rotation (Image.fromarray)

if __name__ == '__main__':
    # path = "tes/before/0001.jpg"
    # image = cv2.imread(path)
    # flip_image(image, 1,path)  # vertical

    image_path ="AutismDataset/test"
    filenames = os.listdir(image_path)
    for filename in filenames:

        # Jumlah Foto test
        jumlah_foto = 150
        category = filename.split('.')[0]
        i = filename.split('.')[1]
        Extension = filename.split('.')[2]

        # index Image
        flip_i = int(i) + jumlah_foto
        rotation_flip_j = flip_i+jumlah_foto
        rotation_k = rotation_flip_j+jumlah_foto

        # Path
        image = cv2.imread(image_path + "/" + filename)

        if category == 'Autistic':
            # Untuk ngetes
            #print(image_path+"/"+filename)
            #print(i + " jadi " + str(flip_i) + " jadi " + str(rotation_flip_j) + " Jadi " + str(rotation_k) +"."+ Extension)

            augmented(image,Extension, 1, image_path, category, flip_i,rotation_flip_j,rotation_k,filename)
        else:
            augmented(image, Extension, 1, image_path, category, flip_i, rotation_flip_j, rotation_k, filename)
