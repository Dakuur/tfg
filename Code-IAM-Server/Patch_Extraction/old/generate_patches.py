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

mrxs_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Raw_WSI"
windows_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches"
fullsize_level = 0
window_size = 256



def readWindow(image, origin, level=0, window_size=256):
    return image.read_region(origin,level,(window_size, window_size))

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
    hospitals = os.listdir("I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/")
    hospitals = [hospital for hospital in hospitals if os.path.isdir("I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/"+hospital)]
    list_hospitals = (["H. Clínic Barcelona"],["H. Broggi", "H. Clínic Barcelona"],["H. Zaragoza"])#
    hospitals = [hospital for hospital in hospitals if hospital in list_hospitals[0]]
    
    for hospital in hospitals:
        images = glob.glob(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/{hospital}/*mask.png")
        
        with open(f"{windows_folder}/{solve(hospital)}/metadata_{solve(hospital)}.csv", "w+") as csv_file:
            csv_file.write("hospital,patient_ID,sample_ID,window_ID,i,j,w,h,infiltration,blurriness,non_white_area\n")
            
            decile_length = len(images)//10
            decile_start = round(decile_length*decile)
            decile_end = round(decile_length*(decile+1))
            if decile == -1:
                decile_start = 0
                decile_end = len(images)
            
            counter = 0
    
            for image in images[decile_start:decile_end]:
                print(f"CURRENT ITER: {counter+1}/{decile_end-decile_start}")
                image_name = image.split("\\")[-1]
                name_data = image_name.split("_")
                hospital = name_data[0]
                if not os.path.exists(f"{windows_folder}/{hospital}"):
                        os.makedirs(f"{windows_folder}/{hospital}")
                patient = name_data[1]
                print(hospital, patient)
                search_result = glob.glob(mrxs_folder+f"/{hospital}/**/{patient}.mrxs", recursive=True)
                if len(search_result)>0:
                    print(image)
                    original_mrxs = search_result[0]
                    mrxs_image = openslide.OpenSlide(original_mrxs)
                    original_size = mrxs_image.level_dimensions[fullsize_level]
                    
                    mask = np.uint8(np.round(io.imread(image)[:,:,0]/128))
                    mask_size = mask.shape
                    try:
                        df_row = df.loc[(df['hospital'] == solve(hospital)) & (df['patient_ID'] == patient)]
                        x_base = df_row["i"].values[-1]
                        y_base = df_row["j"].values[-1]
                        width_base = df_row["w"].values[-1]
                        height_base = df_row["h"].values[-1]
                        size_ratio_base = df_row["size_ratio"].values[-1]
                    except:
                        print(hospital, patient, "Not found in csv")
                        continue
                    
                    #print(original_size)
                    size_ratio = round(height_base/mask_size[0])
                    x_steps = math.floor(width_base/window_size)
                    y_steps = math.floor(height_base/window_size)
                    im = mask
                    im = im>0
                    num_labels, im_comp,stats, _ = cv2.connectedComponentsWithStats(np.uint8(im)*255,connectivity=4)
                    valid_labels = []
                    for nlabel in range(1, num_labels):
                        intersection = np.minimum(im_comp == nlabel, mask==2)
                        #plt.imshow(intersection)
                        #plt.show()
                        #print(np.count_nonzero(intersection))
                        if np.count_nonzero(intersection) > 0:
                            valid_labels.append(nlabel)
                    window_counter = 0
                    createFolder(f"{windows_folder}/{solve(hospital)}/{patient}")
                    with open(f"{windows_folder}/{solve(hospital)}/{patient}/valid_samples.txt", "w+") as txt:
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
                            infiltration_percentage = np.mean(mask[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask])
                            if infiltration_percentage>0:
                                sampleID = getMostCommonValue(im_comp[y_origin_mask:y_end_mask, x_origin_mask:x_end_mask])
                                if not sampleID in valid_labels:
                                    continue
                                window = np.array(readWindow(mrxs_image, (x_origin, y_origin)))[:,:,:3]
                               
                                if np.max(window)<5:
                                    continue
                                if np.min(window)>200:
                                    continue
                                if np.mean(np.std(window, axis=-1)) <15:
                                    continue

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
                                plt.imsave(f"{windows_folder}/{solve(hospital)}/{patient}/{solve(hospital)}_{patient}_{sampleID}_{window_counter}.png", window)
                                csv_file.write(f"{solve(hospital)},{patient},{sampleID},{window_counter},{y_origin},{x_origin},{window_size},{window_size},{max(0,infiltration_percentage-1)},{blurriness},{non_white_area}\n")
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