import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import cv2
from skimage import io
import numpy as np
from matplotlib import pyplot as plt


def solve(name):
    return name.replace("à", "a").replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

rgb_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/RGB_ImagesV3/"
mask_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Segmentation_MasksV3/"
windows_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/PatchesV3/"

rgb_standard_sizes_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/StandardSizedV3/"

sizes = [8192, 4096, 2048, 1024, 512, 256]


def generate_square_image(image, black=False):
	max_side = max(image.shape[:2])
	y_offset = (max_side - image.shape[0])//2
	x_offset = (max_side - image.shape[1])//2
	#print(y_offset, x_offset)
	square_image = None
	if black:
		square_image = np.zeros((max_side, max_side, 3), dtype=np.uint8)
	else:
		square_image = np.ones((max_side, max_side, 3), dtype=np.uint8)*255
	square_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

	#plt.imshow(square_image)
	#plt.show()

	return square_image

hospitals = os.listdir(rgb_folder)
hospital_counter = 0
for hospital in hospitals[hospital_counter:]:
	print(f"{hospital}, {hospital_counter+1}/{len(hospitals)}")
	print("-----------------")
	hospital_counter += 1
	hospital_folder = rgb_folder+solve(hospital)+"/"
	patients = os.listdir(hospital_folder)
	patients = [patient for patient in patients if os.path.isdir(hospital_folder+patient+"/")]
	counter = 0
	for patient in patients:
		print(f"\r{patient}, {counter+1}/{len(patients)}               ", end="")
		counter += 1
		patient_folder = hospital_folder+patient+"/"
		slides = os.listdir(patient_folder)
		for slide in slides:
			slide = slide.split(".png")[0].split("_")[-1]
			if "low" in slide:
				continue
			rgb_path = f"{rgb_folder}{hospital}/{patient}/{hospital}_{slide}.png"
			mask_path = f"{mask_folder}{hospital}/{patient}/{hospital}_{slide}_mask.png"
			if os.path.exists(rgb_path):
				rgb_image = io.imread(rgb_path)[:,:,:3]
				mask_image = io.imread(mask_path)[:,:,:3]
				square_image = generate_square_image(rgb_image)
				
				square_mask = generate_square_image(mask_image, black=True)
				patient_path = f"{rgb_standard_sizes_folder}{solve(hospital)}/{patient}/{slide}" 
				if not os.path.exists(patient_path):
					os.makedirs(patient_path)
				for size in sizes:
					plt.imsave(f"{patient_path}/{solve(hospital)}_{patient}_{slide}_{size}.png", cv2.resize(square_image, (size, size), interpolation=cv2.INTER_AREA))
					plt.imsave(f"{patient_path}/{solve(hospital)}_{patient}_{slide}_mask_{size}.png", cv2.resize(square_mask, (size, size), interpolation=cv2.INTER_AREA))
			else:
				print(f"\nPatient {hospital}/{patient}/{hospital}_{slide}.png not found in RGB Images")
	print()
	print("-----------------")
	print()
                                

