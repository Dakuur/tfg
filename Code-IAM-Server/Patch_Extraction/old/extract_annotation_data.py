import base64
import json

import labelme
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from PIL import Image
OPENSLIDE_PATH = r'C:\Users\pcano\Desktop\CRC_pT1\openslide-win64-20231011\bin'#r'C:\Users\pcano\Desktop\vcpkg-2025.02.14\packages\openslide_x64-windows'#

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
import glob




main_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/"
json_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Annotations/"
mrxs_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Raw_WSI/"
fullsize_level = 0
rgb_level = 3


def readWindow(image, origin, level=0, window_size=256):
    return image.read_region(origin,level,(window_size, window_size))
    #img = readWindow(image, (0,0))
    
                                                                    #(mrxs_cmin,mrxs_rmin), (mrxs_cmax,mrxs_rmax), level=rgb_level
def readSection(image, origin, size, level=0):
    (mrxs_width, mrxs_height)
    size_multiplier = image.level_dimensions[0][0]/image.level_dimensions[level][0]
    x_origin = round(origin[0]/size_multiplier)
    y_origin = round(origin[1]/size_multiplier)
    width = round(size[0]//size_multiplier)
    height = round(size[1]//size_multiplier)
    return np.array(image.get_thumbnail(image.level_dimensions[level]))[y_origin:y_origin+height,x_origin:x_origin+width]#image.read_region(origin,level, size)

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
# load annoated file
files = glob.glob(json_folder+"*Annotated.json")



cases = {}

with open(f"{main_folder}Images/Patient_Images_metadata_all.csv", "w+") as csv_file:
    csv_file.write("hospital,patient_ID,i,j,w,h,size_ratio\n")
    for json_file in files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # load image from data["imageData"]
        original_file = data["imagePath"].replace("_small.png", ".mrxs").replace("_big.png", ".mrxs")
        #print(original_file)
        if len( original_file.split("\\"))<3:
            continue
        hospital = original_file.split("\\")[-2]
        hospital = hospital.replace("H. Marques de Valdecilla","H. M. Valdecilla")
        patient = original_file.split("\\")[-1]
        if "Móstoles" not in hospital:
            continue
        print(hospital,patient)
        case_name = f"{hospital}_{patient}"
        if case_name in cases.keys():
            if "big" in cases[case_name][0]:
                continue
            print("-------------------------")
            print("REPEATED CASE")
            print(case_name)
            print(json_file)
            print(original_file)
            print(cases[case_name])
            cases[case_name].append(json_file)
            print("--------------------------")
        else:
            cases[case_name] = [json_file]
        
        patient = patient.split(".mr")[0]
        original_mrxs_path = mrxs_folder+hospital+"/**/"#+patient#.replace("Hortega Valladolid", "Hortega")
        original_mrxs_path += "".join([letter+"*" for letter in patient[:-1]])
        original_mrxs_path += patient[-1]+".mrxs"
        #print(original_mrxs_path)
        #asgd
        mrxs_path = glob.glob(original_mrxs_path, recursive=True)
        
        '''patient = patient.split(".mr")[0]
        original_mrxs_path = mrxs_folder+hospital+"/**/"#+patient#.replace("Hortega Valladolid", "Hortega")
        original_mrxs_path += "".join([letter+"*" for letter in patient])
        original_mrxs_path +=".mrxs"
        #print(original_mrxs_path)
        #asgd
        mrxs_pathb = glob.glob(original_mrxs_path, recursive=True)'''
        if len(mrxs_path) == 0:
            print("ERROR WITH FILE", original_mrxs_path, "DOES NOT EXIST")
            print(json_file)
            continue
        #continue
        '''if len(mrxs_path) > 0 and len(mrxs_pathb) > 0:
            if not mrxs_path[0] == mrxs_pathb[0]:
                print("-------------")
                print("ERROR, redo files", hospital, patient)
                print(mrxs_path[0])
                print(mrxs_pathb[0])
                print()
                
        continue'''
        mrxs_path = mrxs_path[0].replace("\\", "/")
        patient = mrxs_path.split("/")[-1].split(".")[0]
        #print(mrxs_path)
        #print(patient)
        #continue
        
        
        
        
        image = labelme.utils.img_b64_to_arr(data["imageData"])
        #plt.imshow(image)
        #plt.show()
        
        #print("image:", image.shape, image.dtype)
        
        # load label_names, label, label_points_xy from data["shapes"]
        unique_label_names = ["_background_"] + sorted(
            set([shape["label"] for shape in data["shapes"]])
        )
        label = np.zeros(image.shape[:2], dtype=np.int32)
        label_names = []
        label_points_xy = []
        mask = np.zeros(image.shape[:2], dtype=np.bool_)
        for shape in data["shapes"]:
            label_id = unique_label_names.index(shape["label"])
            points = shape["points"]
            
            s_mask = labelme.utils.shape_to_mask(
                img_shape=image.shape[:2],
                points=shape["points"],
                shape_type=shape["shape_type"],
            )
            mask = np.maximum(mask, s_mask)
        #plt.imshow(image)
        #plt.show()
        #plt.imshow(mask, cmap='gray')
        #plt.show()
        del label
        del label_id
        
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_shape_mask = image_hsv[:,:,1] > 60#60#image_hsv[:,:,2] < np.max(image_hsv[:,:,2])*0.98     5 manresa, 60 la resta
        image_shape_mask = np.minimum(image_shape_mask, image_hsv[:,:,2] > 60)
        #plt.imsave("test.png",image_hsv)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        image_shape_mask = cv2.dilate(np.uint8(image_shape_mask)*255, kernel)
        image_shape_mask = cv2.erode(image_shape_mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
        #mask = cv2.dilate(mask, kernel)
        image_shape_mask = cv2.erode(image_shape_mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        #mask = cv2.dilate(mask, kernel)
        image_shape_mask = cv2.dilate(image_shape_mask, kernel, iterations=2)
        #image_shape_mask = cv2.erode(np.uint8(image_shape_mask)*255, kernel)
        #image_shape_mask = cv2.dilate(image_shape_mask, kernel)
        
        image_mask = (image_shape_mask/255 + mask) * (image_shape_mask/255)
        
        del mask
        del s_mask
        del image_shape_mask
        del image_hsv
        #del image
        
        if np.count_nonzero(image_mask) == 0:
            continue
        rows = np.any(image_mask, axis=1)
        cols = np.any(image_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        #print(rmin, rmax, cmin, cmax)
        
        halfWindowSize = 128
        rmin -= halfWindowSize
        cmin -= halfWindowSize
        rmin = max(0, rmin)
        cmin = max(0, cmin)
        
        rmax += halfWindowSize
        cmax += halfWindowSize
        rmax = min(rmax, image_mask.shape[0])
        cmax = min(cmax, image_mask.shape[1])
        width = cmax-cmin
        height = rmax-rmin
        
        
        
        mrxs_image = openslide.OpenSlide(mrxs_path)
        original_size = mrxs_image.level_dimensions[fullsize_level]
        size_ratio = round(original_size[1]/image_mask.shape[0])
        
        mrxs_rmin = round(rmin*size_ratio)
        mrxs_rmax = round(rmax*size_ratio)
        mrxs_cmin = round(cmin*size_ratio)
        mrxs_cmax = round(cmax*size_ratio)
        
        mrxs_width = mrxs_cmax - mrxs_cmin
        mrxs_height = mrxs_rmax - mrxs_rmin
    
        #print(original_size)
        createFolder(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_Images/{hospital}")
        plt.imsave(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_Images/{hospital}/{hospital}_{patient}_low.png",image)
        try:
            mrxs_section = readSection(mrxs_image, (mrxs_cmin,mrxs_rmin), (mrxs_width, mrxs_height), level=rgb_level)#(mrxs_width,mrxs_height))
            #print(rmin, rmax, cmin, cmax)
            #print(mrxs_rmin, mrxs_rmax, mrxs_cmin, mrxs_cmax)
            #print(mrxs_width, mrxs_cmax-mrxs_cmin)
            #print(mrxs_height, mrxs_rmax-mrxs_rmin)
            #print(mrxs_section.shape)
            #print("-------------")
            #createFolder(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_Images/{hospital}")
            #if not os.path.exists(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_Images/{hospital}"):
            #    os.makedirs(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_Images/{hospital}")
            plt.imsave(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_Images/{hospital}/{hospital}_{patient}.png",mrxs_section)
            del mrxs_section
        except:
            print(f"Couldn't open image in memory for {case_name}")
        #asf
        
        #plt.imshow(image_mask[rmin:rmax, cmin:cmax], cmap='gray')
        #plt.show()
        
        createFolder(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/{hospital}")
        #if not os.path.exists(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/{hospital}"):
            
        #    os.makedirs(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/{hospital}")
        plt.imsave(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/{hospital}/{hospital}_{patient}_mask.png", image_mask[rmin:rmax, cmin:cmax], cmap='gray', vmin=0, vmax=2)
        #plt.imsave(f"I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_Masks/{hospital}/{hospital}_{patient}_mask.png", image[rmin:rmax, cmin:cmax])
        csv_file.write(f"{hospital},{patient},{mrxs_cmin},{mrxs_rmin},{mrxs_width},{mrxs_height},{size_ratio}\n")
        del image_mask
            
        