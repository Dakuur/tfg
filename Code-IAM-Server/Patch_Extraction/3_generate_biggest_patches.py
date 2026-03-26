OPENSLIDE_PATH = r'C:\Users\pcano\Desktop\CRC_pT1\openslide-win64-20231011\bin'

import os
import argparse
#import openslide

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        #print("a<sdg")
        import openslide
else:
    import openslide
import glob
from skimage import io
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image 
import cv2
import pandas as pd
import cv2
import color_correction

mrxs_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Raw_WSI"
RGB_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/WSI_RGB"
standard_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/StandardSized"
masks_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks"
fullsize_level = 0
window_size = 20000


def getMostCommonValue(image):
    values, counts = np.unique(image.flatten(), return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def correctColor(color_matcher, input, target, target_value):
    img_res = color_matcher.transfer(src=input, ref=target, method='MVGD')
    img_res = Normalizer(img_res).uint8_norm()

    img_res_hsv = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)
    value_original = target_value
    value_res = getMostCommonValue(img_res_hsv[:,:,2])
    ratio = float(value_original) / float(value_res)
    img_new_value = img_res_hsv[:,:,2] * ratio
    img_new_value[img_new_value>255] = 255
    img_new_value = np.uint8(img_new_value)

    img_res_hsv[:,:,2] = img_new_value
    img_new = cv2.cvtColor(img_res_hsv, cv2.COLOR_HSV2RGB)

    return img_new


def readWindow(image, origin, level=0, window_size=(256,256)):
    try:
        return image.read_region(origin,level,(window_size[0], window_size[1]))
    except Exception as e:
        print(f"\r Couldn't open patch, {e}", end="")
        return np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)

def getMostCommonValue(image):
    values, counts = np.unique(image, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]  

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def solve(name):
    return name.replace("à", "a").replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")


def generate_patches(decile):
    
    df = pd.read_csv("I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patient_Images_metadata.csv", encoding='utf-8')
    #decile_name = str(decile)
    #if decile == -1:
    #    decile_name = "all"
    hospitals = os.listdir(masks_folder)
    hospitals = [hospital for hospital in hospitals if os.path.isdir(f"{masks_folder}/"+hospital)]
    list_hospitals = (['Consorci Sanitari de Terrassa', 'Cribado Pais Vasco', 'H. Alicante','H. Ourense', 'H. Palamos', 'H. Zaragoza'], ['H. Althaia Manresa', 'H. Basurto', 'H. Bellvitge', 'H. Parc Tauli', 'H. Puerta del Hierro', 'IVO'], ['H. Broggi', 'H. Clinic Barcelona', 'H. Clinico Valencia', 'H. Ramon y Cajal', 'H. Rio Hortega'], ['H. Donostia', 'H. Granollers', 'H. Inca', 'H. Santos Rey', 'H. Tenerife'], [ 'H. M. Valdecilla', 'H. Mostoles', 'H. Murcia',  'H. V. Rocio Sevilla', "H. Vall d'Hebron"])#
    hospitals = [hospital for hospital in hospitals if hospital in list_hospitals[1]+list_hospitals[2]+list_hospitals[3]+list_hospitals[4]]
    
    color_template = io.imread("color_template.png")[:,:,:3]

    for hospital in hospitals[0:]:
            

        patients = os.listdir(f"{masks_folder}/{hospital}/")
        patients = [patient for patient in patients if os.path.isdir(f"{masks_folder}/{hospital}/"+patient)]
        images = glob.glob(f"{masks_folder}/{hospital}/*/*mask.png")
            
        decile_length = len(images)//10
        decile_start = round(decile_length*decile)
        decile_end = round(decile_length*(decile+1))
        if decile == -1:
            decile_start = 0
            decile_end = len(images)
        counter = 0
        for patient in patients:
            images = glob.glob(f"{masks_folder}/{hospital}/{patient}/*mask.png")
    
            for image in images[:]:#decile_start:decile_end]:
                print(f"CURRENT ITER: {counter+1}/{decile_end-decile_start}")
                image_name = image.split("\\")[-1]
                name_data = image_name.split("_")
                #hospital = name_data[0]
                '''if not os.path.exists(f"{windows_folder}/{solve(hospital)}/{patient}"):
                        os.makedirs(f"{windows_folder}/{solve(hospital)}/{patient}")
                if not os.path.exists(f"{cc_folder}/{solve(hospital)}/{patient}"):
                        os.makedirs(f"{cc_folder}/{solve(hospital)}/{patient}")'''
                slide = name_data[-2]
                print(hospital, patient, slide)
                original_hospital_name = hospital.replace("Clinic", "Clínic").replace("Tauli", "Taulí").replace("Mostoles", "Móstoles").replace("Palamos", "Palamós").replace("Rocio", "Rocío")
                search_result = glob.glob(mrxs_folder+f"/{original_hospital_name}/**/{slide}.mrxs", recursive=True)
                print(search_result, mrxs_folder+f"/{hospital}/**/{slide}.mrxs")
                if len(search_result)>0:
                    print(image)
                    original_mrxs = search_result[0]
                    mrxs_image = openslide.OpenSlide(original_mrxs)
                    original_size = mrxs_image.level_dimensions[fullsize_level]

                
                    standard_sized_image = io.imread(standard_folder+f"/{hospital}/{patient}/{slide}/{hospital}_{patient}_{slide}_1024.png")[:,:,:3]
                    transform_data = color_correction.calculateTransformMatrix(standard_sized_image, color_template)
                    
                    try:
                        df_row = df.loc[(df['hospital'] == solve(hospital)) & (df['patient_ID'] == patient) & (df['slide_ID'] == slide)]
                        x_base = df_row["j"].values[-1]
                        y_base = df_row["i"].values[-1]
                        width_base = df_row["w"].values[-1]
                        height_base = df_row["h"].values[-1]
                        size_ratio_base = df_row["size_ratio"].values[-1]
                    except:
                        print(hospital, patient, slide, "Not found in csv")
                        continue
                    
                    #print(original_size)
                    x_steps = math.ceil(width_base/window_size)
                    y_steps = math.ceil(height_base/window_size)
                    max_width = x_base+width_base
                    max_height = y_base+height_base
                    mask = np.uint8(np.round(io.imread(image)/128)[:,:,:3])
                    im = mask[:,:,0]
                    im = im>0
                    affected_mask = np.uint8(mask==2)
                    unified_mask = np.max(affected_mask, axis=-1)
                    im = np.uint8(im)#*255
                    num_labels, im_comp,stats, _ = cv2.connectedComponentsWithStats(np.uint8(im)*255,connectivity=4)
                    valid_labels = []
                    im = np.uint8(im)#*255
                    for nlabel in range(1, num_labels):
                        intersection = np.minimum(im_comp == nlabel, unified_mask>0)
                        #plt.imshow(intersection)
                        #plt.show()
                        #print(np.count_nonzero(intersection))
                        if np.count_nonzero(intersection) > 0:
                            valid_labels.append(nlabel)


                    size_ratio = round(height_base/mask.shape[0])
                    createFolder(f"{RGB_folder}/{solve(hospital)}/{patient}/{slide}")
                    print()
                    for y in range(y_steps):
                        for x in range(x_steps):
                            y_origin = y*window_size + y_base
                            y_end = min(y_origin+window_size, max_height)
                            x_origin = x*window_size + x_base
                            x_end = min(x_origin+window_size, max_width)
                            width = x_end - x_origin
                            height = y_end - y_origin
                            y_origin_mask = round((y*window_size)/size_ratio)
                            y_end_mask = round(((y+1)*window_size)/size_ratio)
                            x_origin_mask = round((x*window_size)/size_ratio)
                            x_end_mask = round(((x+1)*window_size)/size_ratio)
                            y_end = min(y_end_mask, im.shape[0])
                            x_end_mask = min(x_end_mask, im.shape[1])
                            tissue_percentage = np.mean(im[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask])
                            
                            if tissue_percentage>0.05:
                                sampleID = getMostCommonValue(im_comp[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask])
                                if not sampleID in valid_labels:
                                    continue
                                print(f"\rX: {x_origin}/{max_width} ({x}/{x_steps}), Y: {y_origin}/{max_height} ({y}/{y_steps})                         ", end="", flush=True)#

                                window = np.array(readWindow(mrxs_image, (x_origin, y_origin), window_size=(width, height)))[:,:,:3]
                                plt.imsave(f"{RGB_folder}/{solve(hospital)}/{patient}/{slide}/{solve(hospital)}_{patient}_{slide}_{y}_{x}.jpg", window)
                                color_corrected = color_correction.correctImage(window, transform_data)#correctColor(color_matcher, window, color_template, value_original)
                                plt.imsave(f"{RGB_folder}/{solve(hospital)}/{patient}/{slide}/{solve(hospital)}_{patient}_{slide}_{y}_{x}_color_corrected.jpg", color_corrected)
                    print()
    
                counter += 1
                        


def parse_args():
    parser = argparse.ArgumentParser()

    # Results options
    parser.add_argument('--decile', type=int, default=-1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    decile = args.decile
    generate_patches(decile)