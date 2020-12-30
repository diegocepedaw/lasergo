from image_slicer import slice
import os

directory = 'unsliced_textures'
for entry in os.scandir(directory):
    if entry.is_file():
        slice(entry.path, 4)