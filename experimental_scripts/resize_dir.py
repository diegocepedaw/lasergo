import PIL
import os
import os.path
from PIL import Image

f = r'training_images\black_stone'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((30,30))
    img.save(f_img)