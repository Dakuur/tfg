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
windows_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches"
cc_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches_ColorCorrected"
masks_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks"
standard_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/StandardSized"
fullsize_level = 0
window_size = 256


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


def readWindow(image, origin, level=0, window_size=256):
    try:
        return image.read_region(origin,level,(window_size, window_size))
    except Exception as e:
        print(f"\r Couldn't open patch, {e}", end="")
        return np.zeros((window_size, window_size, 3), dtype=np.uint8)

def getMostCommonValue(image):
    values, counts = np.unique(image, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]  

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def solve(name):
    return name.replace("à", "a").replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

def getBlurriness(image, size=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
    (h, w) = gray_image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
	# compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
    fft = np.fft.fft2(gray_image)
    fftShift = np.fft.fftshift(fft)
    
    
    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

	# compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon) + 1e-8)
    mean = np.mean(magnitude)   
    return mean


def generate_patches(decile):
    
    df = pd.read_csv("I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patient_Images_metadata.csv", encoding='utf-8')
    #decile_name = str(decile)
    #if decile == -1:
    #    decile_name = "all"
    hospitals = os.listdir(masks_folder)
    hospitals = [hospital for hospital in hospitals if os.path.isdir(f"{masks_folder}/"+hospital)]
    list_hospitals = (['Consorci Sanitari de Terrassa', 'Cribado Pais Vasco', 'H. Alicante','H. Ourense', 'H. Palamos', 'H. Zaragoza'], ['H. Althaia Manresa', 'H. Basurto', 'H. Bellvitge', 'H. Parc Tauli', 'H. Puerta del Hierro', 'IVO'], ['H. Broggi', 'H. Clinic Barcelona', 'H. Clinico Valencia', 'H. Ramon y Cajal', 'H. Rio Hortega'], ['H. Donostia', 'H. Granollers', 'H. Inca', 'H. Santos Rey', 'H. Tenerife'], [ 'H. M. Valdecilla', 'H. Mostoles', 'H. Murcia',  'H. V. Rocio Sevilla', "H. Vall d'Hebron"])#
    hospitals = [hospital for hospital in hospitals if hospital in list_hospitals[1]]
    
    color_template = io.imread("color_template.png")[:,:,:3]

    for hospital in hospitals[0:]:
        if not os.path.exists(f"{windows_folder}/{solve(hospital)}"):
            os.makedirs(f"{windows_folder}/{solve(hospital)}")
            
        with open(f"{windows_folder}/{solve(hospital)}/metadata_{solve(hospital)}.csv", "w+") as csv_file:
            csv_file.write("hospital,patient_ID,slide_ID,section_ID,window_ID,i,j,w,h,affected_percentage,infiltration,displasia,infiltratrion_and_displasia,blurriness,non_white_area,window_max_value,window_min_value,window_std\n")

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
                
                '''decile_length = len(images)//10
                decile_start = round(decile_length*decile)
                decile_end = round(decile_length*(decile+1))
                if decile == -1:
                    decile_start = 0
                    decile_end = len(images)'''
                
                
        
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

                    
                        #standard_sized_image = io.imread(standard_folder+f"/{hospital}/{patient}/{slide}/{hospital}_{patient}_{slide}_1024.png")[:,:,:3]
                        #transform_data = color_correction.calculateTransformMatrix(standard_sized_image, color_template)
                        
                        mask = np.uint8(np.round(io.imread(image)/128)[:,:,:3])
                        affected_mask = np.uint8(mask==2)
                        unified_mask = np.max(affected_mask, axis=-1)
                        mask_size = mask.shape
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
                        size_ratio = round(height_base/mask_size[0])
                        x_steps = math.floor(width_base/window_size)
                        y_steps = math.floor(height_base/window_size)
                        im = mask[:,:,0]
                        im = im>0
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
                        window_counter = 0
                        createFolder(f"{windows_folder}/{solve(hospital)}/{patient}/{slide}")
                        createFolder(f"{cc_folder}/{solve(hospital)}/{patient}/{slide}")
                        with open(f"{windows_folder}/{solve(hospital)}/{patient}/{slide}/valid_samples.txt", "w+") as txt:
                            if len(valid_labels)>0:
                                txt.write(str(valid_labels[0]))
                                for i in range(1,len(valid_labels)):
                                    txt.write(f",{valid_labels[i]}")
                        #continue
                        for y in range(y_steps):
                            for x in range(x_steps):
                                y_origin = y*window_size + y_base
                                y_end = y_origin+window_size
                                x_origin = x*window_size + x_base
                                x_end = x_origin+window_size
                                
                                y_origin_mask = round((y*window_size)/size_ratio)
                                y_end_mask = round(((y+1)*window_size)/size_ratio)
                                x_origin_mask = round((x*window_size)/size_ratio)
                                x_end_mask = round(((x+1)*window_size)/size_ratio)
                                tissue_percentage = np.mean(im[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask])
                                
                                if tissue_percentage>0:
                                    sampleID = getMostCommonValue(im_comp[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask])
                                    if not sampleID in valid_labels:
                                        continue
                                    window = np.array(readWindow(mrxs_image, (x_origin, y_origin)))[:,:,:3]
                                    w_max = np.max(window)
                                    w_min = np.min(window)
                                    w_std = np.mean(np.std(window, axis=-1))
                                    '''if w_max<5:
                                        continue
                                    if w_min>200:
                                        continue
                                    if w_std <15:
                                        continue'''
                                    infiltration_percentage = np.mean(affected_mask[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask, 0]) / tissue_percentage
                                    both_percentage = np.mean(affected_mask[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask, 1]) / tissue_percentage
                                    displasia_percentage = np.mean(affected_mask[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask, 2]) / tissue_percentage
                                    affected_percentage =  np.mean(unified_mask[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask]) / tissue_percentage


                                    non_white_area = np.minimum(np.minimum(np.max(window, axis=-1)>=5,np.max(window, axis=-1)<=200),np.std(window, axis=-1)>=15)
                                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
                                    image_shape_mask = cv2.dilate(np.uint8(non_white_area)*255, kernel, iterations=2)
                                    image_shape_mask = cv2.erode(image_shape_mask, kernel)
                                    non_white_area = np.count_nonzero(image_shape_mask)/(window_size**2)
                                    #print(non_white_area)
                                    #plt.imshow(window)
                                    #plt.show()
                                    #plt.imshow(image_shape_mask)
                                    #plt.show()
                                    blurriness = getBlurriness(window)
                                    plt.imsave(f"{windows_folder}/{solve(hospital)}/{patient}/{slide}/{solve(hospital)}_{patient}_{slide}_{sampleID}_{window_counter}.png", window)
                                    color_corrected = color_correction.correctImage(window, transform_data)#correctColor(color_matcher, window, color_template, value_original)
                                    plt.imsave(f"{cc_folder}/{solve(hospital)}/{patient}/{slide}/{solve(hospital)}_{patient}_{slide}_{sampleID}_{window_counter}_color_corrected.png", color_corrected)
                                    csv_file.write(f"{solve(hospital)},{patient},{slide},{sampleID},{window_counter},{y_origin},{x_origin},{window_size},{window_size},{affected_percentage},{infiltration_percentage},{displasia_percentage},{both_percentage},{blurriness},{non_white_area},{w_max},{w_min},{w_std}\n")
                                    window_counter += 1
        
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