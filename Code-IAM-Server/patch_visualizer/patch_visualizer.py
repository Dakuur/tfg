import glob
import PySimpleGUI as sg
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import pandas as pd

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    #plt.close('all')
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


#patient_data_layout = [sg.Text("Patient ID: "),sg.In(size=(25, 1), enable_events=True, key="-PATIENT ID INPUT-"), sg.Text("", key="-PATIENT ID LABEL-")]


hospital = "H. Clinic Barcelona"
main_folder = f"../../Images/Patches/{hospital}/"

max_elements = 200


folders= sorted(glob.glob(main_folder+"*/"))
folders = [folder.replace("\\","/").split("/")[-2] for folder in folders]

#print(folders)
#testOpenAll(folders, mainFolder)

users_layout = [sg.Text("Usuaris\t\t\t"), sg.Text("Pacients infiltrats\t\t\t"), sg.Text("Pacients no infiltrats")],[sg.Listbox( values=folders, enable_events=True, size=(25, 20), key="-PATIENT LIST-"), sg.Listbox( values=[], enable_events=True, size=(25, 20), key="-POSITIVES LIST-"), sg.Listbox( values=[], enable_events=True, size=(25, 20), key="-NEGATIVES LIST-"), sg.Canvas(key='-POSITIVE-', size=(512,512)), sg.Push(), sg.Canvas(key='-NEGATIVE-', size=(512,512))]

#frameButtonsLayout = [sg.Text("Infiltracio positiu:"),sg.Text("", key="-POSITIVE-TEXT-"),sg.Text("Infiltracio negatiu:"),sg.Text("", key="-NEGATIVE-TEXT-")]
layout = [users_layout]#,
#frameButtonsLayout]
current_patient = ""

hospital_df = pd.read_csv(f"{main_folder}metadata_{hospital}.csv")
patient_df = None

px = 1/plt.rcParams['figure.dpi']

figRGBneg, axRGBneg = plt.subplots(1,1, figsize = (256*px, 256*px))
figRGBneg_canvas_agg = None
figRGBpos, axRGBpos = plt.subplots(1,1, figsize = (256*px, 256*px))
figRGBpos_canvas_agg = None

def updatePosImage(current_patient, current_positive):
	global figRGBpos, axRGBpos, figRGBpos_canvas_agg
	#axRGBpos.cla()
	#figRGBpos, axRGBpos = plt.subplots(1,1)#, figsize = (256*px, 256*px))
	positive_sample, positive_window, positive_infiltration = current_positive.split("_")
	current_positive_file = f"{hospital}_{current_patient}_{positive_sample}_{positive_window}"
	
	null_image = np.zeros((9,9,3), dtype=np.uint8)
	for i in range(9):
		null_image[i,0+i%2::2,:] = 127
	RGBposimage = null_image[:,:,:]

	RGBpos_path = f"{main_folder}{current_patient}/{current_positive_file}.png"
	if os.path.exists(RGBpos_path):
		RGBposimage = io.imread(RGBpos_path)[:,:,:3]
	else:
		print("POSITIVE DOES NOT EXIST")

	
	#axRGBpos.cla()
	axRGBpos.imshow(RGBposimage)

	axRGBpos.set_title(f"Infiltracio: {positive_infiltration}")
	if figRGBpos_canvas_agg:
		figRGBpos_canvas_agg
	else:
		figRGBpos_canvas_agg = draw_figure(window['-POSITIVE-'].TKCanvas, figRGBpos)
	#	delete_figure_agg(figRGBpos_canvas_agg)
	figRGBpos_canvas_agg.draw()
	
	#figRGBpos_canvas_agg = draw_figure(window['-POSITIVE-'].TKCanvas, figRGBpos)
	#window['-POSITIVE-'].set_size((256, 256))

	return

def updateNegImage(current_patient, current_negative):
	global figRGBneg, axRGBneg, figRGBneg_canvas_agg
	#axRGBneg.cla()
	#figRGBneg, axRGBneg = plt.subplots(1,1)#, figsize = (256*px, 256*px))

	negative_sample, negative_window, negative_infiltration = current_negative.split("_")
	current_negative_file = f"{hospital}_{current_patient}_{negative_sample}_{negative_window}"
	
	null_image = np.zeros((9,9,3), dtype=np.uint8)
	for i in range(9):
		null_image[i,0+i%2::2,:] = 127
	RGBnegimage = null_image[:,:,:]


	RGBneg_path = f"{main_folder}{current_patient}/{current_negative_file}.png"
	if os.path.exists(RGBneg_path):
		RGBnegimage = io.imread(RGBneg_path)[:,:,:3]
	else:
		print("NEGATIVE DOES NOT EXIST")

	#axRGBneg.cla()
	axRGBneg.imshow(RGBnegimage)
	axRGBneg.set_title(f"Infiltracio: {negative_infiltration}")
	if figRGBneg_canvas_agg:
		figRGBneg_canvas_agg
	else:
		figRGBneg_canvas_agg = draw_figure(window['-NEGATIVE-'].TKCanvas, figRGBneg)
	#	delete_figure_agg(figRGBneg_canvas_agg)
	#figRGBneg_canvas_agg = draw_figure(window['-NEGATIVE-'].TKCanvas, figRGBneg)
	#window['-NEGATIVE-'].set_size((256, 256))
	figRGBneg_canvas_agg.draw()
	return

window = sg.Window("Patch Visualizer", layout, size=(1920,1080), return_keyboard_events=True, use_default_focus=False, finalize=True, resizable=True)
#window['-SET FRAME-'].bind("<Return>", "_Enter")


current_positive = "a_a_a"
current_negative = "a_a_a"


#
while True:
	event, values = window.read()#timeout=100
	# End program if user closes window or
	# presses the OK button
	if event == sg.WIN_CLOSED:
		break		

	elif event == "-PATIENT LIST-":
		current_patient = values["-PATIENT LIST-"][0]
		csv_path = "dfs/"+current_patient+".csv"
		if os.path.exists(csv_path):
			patient_df = pd.read_csv(csv_path)
		else:
			patient_df = hospital_df[(hospital_df['hospital'] == hospital) & (hospital_df['patient_ID'] == current_patient)]#[:max_elements]
			patient_df.to_csv(csv_path)
		positives_df = patient_df[(patient_df['infiltration'] > 0.9)]
		negatives_df = patient_df[(patient_df['infiltration'] < 0.1)]

		positives_list = positives_df["sample_ID"].astype(str) + "_" + positives_df["window_ID"].astype(str) + "_" + positives_df["infiltration"].astype(str)
		negatives_list = negatives_df["sample_ID"].astype(str) + "_" + negatives_df["window_ID"].astype(str) + "_" + negatives_df["infiltration"].astype(str)



		#print(patients)
		window["-POSITIVES LIST-"].update(positives_list.values)
		window["-NEGATIVES LIST-"].update(negatives_list.values)
		if len(positives_list.values) > 0:
			current_positive = positives_list.values[0]
		if len(negatives_list.values) > 0:
			current_negative = negatives_list.values[0]
		
		updatePosImage(current_patient, current_positive)
		updateNegImage(current_patient, current_negative)

	elif event == "-POSITIVES LIST-":
		current_positive = values["-POSITIVES LIST-"][0]
		updatePosImage(current_patient, current_positive)

	elif event == "-NEGATIVES LIST-":
		current_negative = values["-NEGATIVES LIST-"][0]
		updateNegImage(current_patient, current_negative)
		

	

window.close()
