import glob
import pandas as pd
import os
import shutil

windows_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches/"

hospitals = os.listdir(windows_folder)
hospitals = [hospital for hospital in hospitals if os.path.isdir( windows_folder + hospital)]
#print(hospitals)
hospitals = [hospital for hospital in hospitals if (("í" in hospital) or ("á" in hospital) or ("é" in hospital) or ("ó" in hospital) or ("ú" in hospital) or ("à" in hospital) or ("è" in hospital) or ("ì" in hospital) or ("ò" in hospital) or ("ù" in hospital))]
print(hospitals)
#hospitals = [hospital for hospital in hospitals if not "Vasco" in hospital]
#print(hospitals)

def solve(name):
	return name.replace("à", "a").replace("è", "e").replace("ì", "i").replace("ò", "o").replace("ù", "u").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")

for hospital in hospitals[:]:
	new_hospital_name = solve(hospital)
	print()
	print(f"------------------------------------------")
	print(f"--------------{hospital}--------------")
	hospital_folder = windows_folder + hospital + "/"
	hospital_csv_path = hospital_folder+"metadata_"+hospital+".csv"
	data = []
	if os.path.exists(hospital_csv_path):
		with open(hospital_csv_path, "r") as textfile:
			data = textfile.readlines()
		with open(hospital_csv_path, "w+") as textfile:
			for datum in data:
				textfile.write(solve(datum))

		os.rename(hospital_csv_path, hospital_folder+"metadata_"+new_hospital_name+".csv")

	patients = os.listdir(hospital_folder)
	patients = [patient for patient in patients if not "." in patient]
	#flag = False
	for patient in patients:
		#if "18-930-G1" in patient:
		#	flag = True
		#if not flag:
		#	continue
		print(f"************{patient}***********")
		patient_folder = hospital_folder + patient + "/"
		
		patches = glob.glob(patient_folder+"*.png")
		
		for patch in patches:
			patch_name = patch.split("\\")[-1]
			new_patch_name = solve(patch_name)
			os.rename(patient_folder+patch_name, patient_folder+new_patch_name)
		
		print()
	os.rename(windows_folder+hospital, windows_folder+new_hospital_name)
	#break

	###afegir neteja excel


