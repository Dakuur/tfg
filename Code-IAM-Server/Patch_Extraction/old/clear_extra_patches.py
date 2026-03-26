import glob
import pandas as pd
import os
import shutil

windows_folder = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Images/Patches/"
valid_slides_path = "I:/Database/MedicalImaging/HistoPatologia/ColonCancer/PrivateBD/PEARSON/Code/Patch_Extraction/valid_slides.txt"

valid_slides = []
with open(valid_slides_path, "r") as txt:
	valid_slides = txt.readline().split(",")

hospitals = os.listdir(windows_folder)
hospitals = [hospital for hospital in hospitals if os.path.isdir( windows_folder + hospital)]
#hospitals = [hospital for hospital in hospitals if not "Vasco" in hospital]
#print(hospitals)

for hospital in hospitals:#[9:-3]:
	print()
	print(f"------------------------------------------")
	print(f"--------------{hospital}--------------")
	hospital_folder = windows_folder + hospital + "/"
	hospital_csv_path = hospital_folder+"metadata_"+hospital+".csv"
	hospital_df = pd.read_csv(hospital_csv_path, encoding='latin1')
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
		if not patient in valid_slides:
			print("patient to delete", hospital,patient)
			shutil.rmtree(patient_folder)
			continue
		
		patches = glob.glob(patient_folder+"*.png")
		valid_sections = []
		with open(patient_folder+"valid_samples.txt", "r") as txt:
			valid_sections = txt.readline().split(",")
			if not valid_sections[0]=="":
				valid_sections = [int(valid) for valid in valid_sections]
			else:
				print("patient to delete", hospital,patient)
				try:
					shutil.rmtree(patient_folder)
				except:
					print(f"Couldn't remove {patient_folder}")
				continue

		#print(valid_sections)
		patient_df = hospital_df[(hospital_df['hospital'] == hospital) & (hospital_df['patient_ID'] == patient)]
		#print(patient_df)
		#asgd
		df_row = None
		for patch in patches:
			patch_name = patch.split("\\")[-1].split(".")[-2]
			patch_data = patch_name.split("_")
			print(f"\r {patch_name}     {patch_data}        ", end="")
			section = int(patch_data[-2])
			window = int(patch_data[-1])

			if section not in valid_sections:
				#print(hospital, patient, patch_name)
				#print(section, valid_sections)
				#print("should delete")
				os.remove(patch)
			else:
				#pass
				df_row = patient_df.loc[(patient_df['sample_ID'] == section) & (patient_df['window_ID'] == window)]
				if len(df_row.values) == 0:
					os.remove(patch)
		
		print()
	#break

	###afegir neteja excel


