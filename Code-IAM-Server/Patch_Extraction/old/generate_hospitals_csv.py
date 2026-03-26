import pandas as pd 

correct_data = pd.read_excel("../../pT1 CRC CASOS DEFINITIUS AMB ITEMS HISTOLOGICS CONSENSUS.xlsx")
#print(correct_data.columns)
slides_values = correct_data["Annotated slide "].values
valid_slides = []
for slide in slides_values:
	for item in slide.replace(" ","").split(","):
		valid_slides.extend(item.split(";"))
with open("valid_slides.txt", "w+") as txt:
	txt.write(valid_slides[0])
	for i in range(1,len(valid_slides)):
		txt.write(f",{valid_slides[i]}")

current_hospital ="hospital"
current_file = open("test.txt", "w+", encoding="utf-8")
full_data = open("../../Images/Patches/Pearson_metadata.csv", "r", encoding="utf-8")
for line in full_data:
	data = line.split(",")
	hospital = data[0]
	slide_id = data[1]
	if not hospital == current_hospital:
		print(hospital)
		current_hospital = hospital
		current_file.close()
		current_file = open(f"../../Images/Patches/{hospital}/metadata_{hospital}.csv", "w+", encoding="utf-8")
		current_file.write('hospital,patient_ID,sample_ID,window_ID,i,j,w,h,infiltration\n')
	if slide_id in valid_slides:
		current_file.write(line)
		print(line[:-2] +" valid")



current_file.close()