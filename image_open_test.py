from skimage import io
import os

images_dir = '/home/Biao/data/images__all/'

for image_name in os.listdir(images_dir):
    try:
        img_name = os.path.join(images_dir,image_name)
        image = io.imread(img_name)
    except:
        print(image_name)