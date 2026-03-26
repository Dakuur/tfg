OPENSLIDE_PATH = '../openslide-win64-20231011/bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import glob


paths = glob.glob("D:/CRC_Data/*Broggi*/*/*/*.mrxs") #All .mrxs, can be filtered by hospital: Broggi, Barcelona, Basurto, etc
print(paths)
for path in paths:
    path = path.replace("\\", "/")
    print(path)
    image = openslide.OpenSlide(path)
    
    folder = "results/"+path.split("/")[-4] #create a folder for each hospital
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    #generate WSI at 1/2^4 size (BIG)
    img = image.get_thumbnail(image.level_dimensions[4])
    img.save("results/"+path.split("/")[-4]+"/"+path.split("/")[-1][:-5]+"_big.png", "PNG")
    
    #generate WSI at 1/2^6 size (SMALL)
    img=image.get_thumbnail(image.level_dimensions[6])
    img.save("results/"+path.split("/")[-4]+"/"+path.split("/")[-1][:-5]+"_small.png", "PNG")