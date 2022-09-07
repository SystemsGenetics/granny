import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import tensorflow as tf
tf.autograph.set_verbosity(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
import pathlib
import shutil
import skimage
import GRANNY_config as config
import re
import matplotlib.pyplot as plt
import cv2
from tkinter import * 
import PIL as pil
from PIL import ImageTk, Image
from tkinter import filedialog

class GRANNY(object): 
	def __init__(self):
		self.ROOT_DIR = pathlib.Path(__file__).parent.resolve()
		self.MODEL_DIR = os.path.join(os.path.curdir, "logs")
		self.NEW_DATA_DIR = "input_data" + os.sep
		# self.PRETRAINED_MODEL = os.path.join(self.ROOT_DIR, "Mask_RCNN-2.1", "mask_rcnn_balloon.h5")
		self.PRETRAINED_MODEL = os.path.join(self.ROOT_DIR, "mask_rcnn_balloon.h5")
		self.VERBOSE = 1
		self.FILE_NAME = ""
		self.FOLDER_NAME = ""
		self.OLD_DATA_DIR = ""
		self.ACTION = ""
		self.FILE_EXTENSION = (
			".JPG", ".JPG".lower(),
			".PNG", ".PNG".lower(),
			".JPEG", ".JPEG".lower(),
			".TIFF", ".TIFF".lower(),
		)
		
		# Location where masked apple trays will be saved
		self.FULLMASK_DIR = "full_masked_data" + os.sep

		# Location where segmented/individual apples will be saved 
		self.SEGMENTED_DIR = "segmented_data" + os.sep

		# Location where apples with the scald removed will be saved
		self.BINARIZED_IMAGE = "binarized_data" + os.sep

		
	def clean_binarized_dir(self): 
		self.check_path(self.BINARIZED_IMAGE, reset = 1)

	def setAction(self, action, fname, mode): 
		"""
			Setter method for action to perform in main()
			--action extract: run mask_extract_image() to extract individual apples from tray
			--action rate: run rate_binarize_image() to obtain ratings for each single-apple image
			--fname: folder name for --action extract
			--fname: file name for --action rate --mode 1
			--fname: folder name for --action rate --mode 2
			--mode 1: single image binarization for --action rate
			--mode 2: multiple image binarization for --action rate 
		"""
		self.ACTION = action
		if mode == None: 
			self.MODE = 1
		else: 
			self.MODE = mode
		if fname == None: 
			self.OLD_DATA_DIR = os.path.curdir
			self.FILE_NAME = os.path.curdir
			self.FOLDER_NAME = os.path.curdir
		else: 
			if self.ACTION == "extract": 
				self.OLD_DATA_DIR = fname
			elif self.ACTION == "rate": 
				self.FILE_NAME = fname
				self.FOLDER_NAME = fname
	

	def setVerbosity(self, verbose):
		""" 
			Setter method for configuration verbosity
			--verbose 1: print configuration summary
		"""
		self.VERBOSE = verbose

	def check_path(self, dir, reset = 0):
		"""
			Check and create a directory
			reset = remove the existing and re-create an empty directory
		"""
		if os.path.exists(dir):
			if reset: 
				shutil.rmtree(dir)
				os.makedirs(dir)
		else:
			os.makedirs(dir)

	def list_all(self, data_dir = os.path.curdir):
		""" 
			List all the folder names and image file names in the directory
		"""
		file_name = []
		folder_name = []
		for root, dirs, files in os.walk(data_dir):
			for file in files:
				if file.endswith(self.FILE_EXTENSION):
					file_name.append(os.path.join(root, file))
			for fold in dirs:
				folder_name.append(os.path.join(root,fold))
		return folder_name, file_name

	def clean_name(self, fname): 
		""" 
			Remove file extensions in file names 
		"""
		for ext in self.FILE_EXTENSION:
			fname = re.sub(ext, "", fname)
		return fname 

	def load_model(self, verbose = 1):
		""" 
			Load pretrained model, download if the model does not exist
		"""
		if not os.path.exists(self.PRETRAINED_MODEL):
			config.MRCNN_utils.download_trained_weights(self.PRETRAINED_MODEL)
		AppleConfig = config.AppleConfig()
		if verbose: 
			AppleConfig.display()
		model = config.MRCNN_model.MaskRCNN(mode = "inference", model_dir = self.MODEL_DIR, config = AppleConfig)
		model.load_weights(self.PRETRAINED_MODEL, by_name=True)
		return model
	
	def create_fullmask_image(self, model, im, fname = ""): 
		""" 
			Apply the model on the image to identify individual apples
			model: Mask-RCNN model 
			im: full image (tray of apples) to mask 
			data_dir: directory to save the masked image 
			fname: file (image) name
		"""
		class_names = ["BG", ""]
		results = model.detect([im], verbose=0)
		r = results[0]
		mask = r["masks"]
		mask = mask.astype(int)
		box  = r["rois"]
		score = r["scores"]
		config.MRCNN_visualize.display_instances(im, box, mask, r['class_ids'], 
								class_names, score)
		plt.savefig(os.path.join(fname + ".png"))
		return mask, box
	
	def sort_apple(self, df): 
		""" 
			Sort apples using their center coordinates stored in df 
			This sorting algorithm follows the numbering convention 
			in '01-input_data/GS-1-16_FilesForImageAnalysis/GS-1-16_ImageTutorial.pptx'
			df: panda DataFrame object containing coordinate information of the detected objects 
		"""
		df = df.sort_values("ycenter", ascending= True, ignore_index = True)
		rows = 1
		count = 0
		df.append(df.iloc[-1])
		for count in range(0, len(df)-1):
			df["rows"].iloc[count] = rows
			if not np.abs(df["ycenter"].iloc[count+1] - df["ycenter"].iloc[count]) < 300: 
				rows += 1
		df["rows"].iloc[-1] = rows
		df_list = []
		if len(df[df["rows"] == 1]) == 5:
			apple_id = 18
			for i in range(1,5):
				dfx = df[df["rows"] == i].sort_values("xcenter", ascending= False, inplace = False, ignore_index = True)
				for id in range(0, len(dfx)): 
					dfx["apple_id"].iloc[id] = apple_id
					apple_id -= 1
				df_list.append(dfx)
		else:
			apple_id = 1
			for i in range(1,5):
				dfx = df[df["rows"] == i].sort_values("xcenter", ascending= False, inplace = False, ignore_index = True)
				for id in range(0, len(dfx)): 
					dfx["apple_id"].iloc[id] = apple_id
					apple_id += 1
				df_list.append(dfx)
		return df_list 

	def process_box(self, box): 
		""" 
			Sort apples
			box: the coordinate information obtained from the Mask-RCNN
		"""
		df = pd.DataFrame(box)
		df.columns = ["y1", "x1", "y2", "x2"]
		df = df.iloc[0:18]
		df["ycenter"] = ((df["y1"]+df["y2"])/2).astype(int)
		df["xcenter"] = ((df["x1"]+df["x2"])/2).astype(int)
		df["rows"] = 0
		df["apple_id"] = 0
		df["nums"] = df.index
		apple_list = self.sort_apple(df)
		return apple_list
	
	def extract_image(self, df_list, mask, im, fname = ""): 
		""" 
			Extract individual image from masks created by Mask-RCNN 
			df_list: sorted list containing position of individual apples
			mask: binary mask of individual apples
			im: full image (tray of apples) to extract from
			data_dir: directory to save the images
			fname: file name 
		"""
		for df in df_list: 
			for i in range(0,len(df)):
				ar = np.array(df)
				m = mask[:,:,ar[i][-1]]
				new_im = np.zeros([ar[i][2]-ar[i][0],ar[i][3]-ar[i][1],3], dtype = np.uint8)
				for j in range(0,im.shape[2]):
					new_im[:,:,j] = im[ar[i][0]:ar[i][2],ar[i][1]:ar[i][3],j]*m[ar[i][0]:ar[i][2],ar[i][1]:ar[i][3]]
				plt.imsave(fname + "_" + str(ar[i][-2]) + ".png", new_im)

	
	def remove_purple(self, img): 
		"""
			Remove the surrounding purple from the green apple
		"""
		new_img = img
		ycc_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)
		channel1Min = 0*bin
		channel1Max = 255*bin
		channel2Min = 0*bin
		channel2Max = 255*bin
		channel3Min = 0*bin
		channel3Max = 126*bin
		th1 = np.greater_equal(ycc_img[:,:,0], channel1Min) & np.less_equal(ycc_img[:,:,0], channel1Max)
		th2 = np.greater_equal(ycc_img[:,:,1], channel2Min) & np.less_equal(ycc_img[:,:,1], channel2Max)
		th3 = np.greater_equal(ycc_img[:,:,2], channel3Min) & np.less_equal(ycc_img[:,:,2], channel3Max)
		th123 = (th1 & th2 & th3) 
		for i in range(3):
			new_img[:,:,i] = new_img[:,:,i] * th123
		return new_img 
	
	def smooth_binary_mask(self, bin_mask): 
		"""
			Smooth scald region with basic morphological operations
			bin_mask: binary mask
		"""
		bin_mask = np.uint8(bin_mask)
		ksize = (20, 20)
		strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize = ksize)
		bin_mask = cv2.dilate(cv2.erode(bin_mask, kernel = strel, iterations = 1), kernel = strel, iterations = 1)
		bin_mask = cv2.erode(cv2.dilate(bin_mask, kernel = strel, iterations = 1), kernel = strel, iterations = 1)
		return bin_mask

	def segment_green(self, img): 
		"""
			Remove the scald region from the single apple image 
			The stem could have potentially been removed, 
		"""
		new_img = img
		lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)
		im_range = np.float32(90)    
		for i in range(3):
			lab_img[:,:,i] = (lab_img[:,:,i] - np.amin(lab_img[:,:,i]))/(np.amax(lab_img[:,:,i])- np.amin(lab_img[:,:,i]))
			lab_img[:,:,i] *= 90
		lab_img = lab_img.astype(np.uint8)
		hist, bin_edges = np.histogram(lab_img[:,:,1], bins = 90)
		bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)
		channel1Min = 1*bin
		channel1Max = im_range*bin
		channel2Min = -1/2*im_range*bin
		channel2Max = (np.argmax(hist) - 2/9*im_range + 1)*bin
		channel3Min = 1*bin
		channel3Max = im_range*bin
		th1 = np.greater_equal(lab_img[:,:,0], channel1Min) & np.less_equal(lab_img[:,:,0], channel1Max)
		th2 = np.greater_equal(lab_img[:,:,1], channel2Min) & np.less_equal(lab_img[:,:,1], channel2Max)
		th3 = np.greater_equal(lab_img[:,:,2], channel3Min) & np.less_equal(lab_img[:,:,2], channel3Max)
		th123 = (th1 & th2 & th3) 
		th123 = self.smooth_binary_mask(th123)
		for i in range(3):
			new_img[:,:,i] = new_img[:,:,i] * th123
		return th123, new_img
	
	def calculate_scald(self, bw, img):
		"""
			Calculate scald region
			bw: binarized image
			img: original image to be used as ground truth
		"""
		fraction = 0 
		img = np.uint8(img)
		ground_area = 1/3*np.count_nonzero(img[:,:,0:2])
		mask_area = 1/3*np.count_nonzero(bw[:,:,0:2])
		fraction = 1 - mask_area/ground_area 
		if fraction < 0: 
			fraction = 0
		return fraction

	def score_image(self, img): 
		""" 
			Clean up individual image (remove purple area of the tray),
			and perform image binarization 
			img: individual apple image 
		"""
		# Resize image to 800x800
		img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
		old_img = img

		# Remove surrounding purple
		img = self.remove_purple(img)
		nopurple_img = img
		
		# Image smoothing
		img = cv2.GaussianBlur(img, (3,3), sigmaX=0, sigmaY=0) 
		
		# Image binarization 
		bw, img = self.segment_green(img)

		return nopurple_img, img, bw

	def rotate_image(self, old_im_dir, new_im_dir = ""):
		"""
			Check and rotate image to landscape (4000 x 6000)
			old_im_dir: directory of the original image
			new_im_dir: directory of the rotated image
		"""
		img = skimage.io.imread(old_im_dir)
		if img.shape[0]>img.shape[1]:
			img = skimage.transform.rotate(img, 90, resize = True)
		skimage.io.imsave(new_im_dir, img)
		return img

	def rate_binarize_image(self): 
		""" 
			Main method performing Image Binarization on individual images 
			Score will be written to a .txt file
		"""
		self.check_path(self.BINARIZED_IMAGE)
		if self.MODE == 1:
			try:
				file_name = self.FILE_NAME
				img = skimage.io.imread(file_name)
				nopurple_img, binarized_image, bw = self.score_image(img)
				score = self.calculate_scald(binarized_image, nopurple_img)
				idx = -file_name[::-1].find(os.sep)
				file_name = file_name[idx:]
				skimage.io.imsave(os.path.join(self.BINARIZED_IMAGE, file_name), binarized_image)
				with open("rating.txt","w") as w:
					w.writelines(f"{self.clean_name(file_name)}:\t\t{score}")
					w.writelines("\n")
			except FileNotFoundError:
				print(f"\t- Folder/File does not exist. -")
		elif self.MODE == 2:
			try:
				folders, files = self.list_all(self.FOLDER_NAME)
				for folder in folders:
					self.check_path(folder.replace(self.FOLDER_NAME, self.BINARIZED_IMAGE))
				scores = []
				for file_name in files: 
						img = skimage.io.imread(file_name)
						file_name = self.clean_name(file_name)
						nopurple_img, binarized_image, bw = self.score_image(img)
						score = self.calculate_scald(binarized_image, nopurple_img)
						idx = -file_name[::-1].find(os.sep)
						file_name = file_name[idx:]
						skimage.io.imsave(os.path.join(self.BINARIZED_IMAGE, file_name + ".png"), binarized_image)
						scores.append(score)
				with open("ratings.txt","w") as w:
					for i, score in enumerate(scores)	: 
						w.writelines(f"{self.clean_name(files[i])}:\t\t{score}")
						w.writelines("\n")
			except FileNotFoundError: 
				print(f"\t- Folder/File Does Not Exist -")
		else: 
			print("-\t Invalid MODE. Specify either \"1\" or \"2\". -")

	def mask_extract_image(self): 
		"""
			Main method performing Image Masking and Image Extraction on full tray images
			Output directory: 'segmented_data' and 'full_masked_data'
				'segmented_data/': contains extracted images of individual apples
				'full_masked_data/': contains masked images of apple trays
			Time: ~ 4-5 minutes per	 full-tray image
		"""
		try: 
			model = self.load_model(verbose = self.VERBOSE)
			data_dirs, file_names = self.list_all(self.OLD_DATA_DIR)
			for data_dir in data_dirs: 
				self.check_path(data_dir.replace(self.OLD_DATA_DIR, self.FULLMASK_DIR))
				self.check_path(data_dir.replace(self.OLD_DATA_DIR, self.SEGMENTED_DIR))
			for file_name in file_names:
				img = self.rotate_image(
								old_im_dir = file_name, 
								new_im_dir = file_name.replace(self.OLD_DATA_DIR, self.NEW_DATA_DIR), 
							)
				file_name = self.clean_name(file_name)
				mask, box = self.create_fullmask_image(
					model = model, 
					im = img,
					fname = file_name.replace(self.OLD_DATA_DIR, self.FULLMASK_DIR)
					)
				if len(box) >= 18:
					df_list = self.process_box(box)
					self.extract_image(
						df_list = df_list,
						mask = mask,
						im = img,
						fname = file_name.replace(self.OLD_DATA_DIR, self.SEGMENTED_DIR)
					)
		except FileNotFoundError:
			print(f"\t- Folder/File Does Not Exist -")

	def launch_gui(self):
		"""
			Launch a GUI zwindow asking for directory name/ file name to perform \"--action\" on
			Currently, the GUI is still under construction
		"""
		win = Tk()
		if self.ACTION == "extract": 
			win.dirname = filedialog.askdirectory(
				initialdir = os.path.curdir,
				message = "Select An Image Directory", 
				mustexist = True,
			)
			self.OLD_DATA_DIR = win.dirname
			self.mask_extract_image()
		elif self.ACTION == "rate": 
			if self.MODE == 1:
				win.filename = filedialog.askopenfile(
					initialdir = os.path.curdir,
					title = "Select An Image", 
					filetypes = [("Image Files",f) for f in self.FILE_EXTENSION],
				)
				self.FILE_NAME = win.filename.name
			elif self.MODE == 2: 
				win.dirname = filedialog.askdirectory(
					initialdir = os.path.curdir,
					message = "Select An Image Directory", 
					mustexist = True,
				)
				self.FOLDER_NAME = win.dirname
			else: 
				print("\t- Invalid MODE. Specify \"2\" for multiple images processing-")
			self.rate_binarize_image()
		else: 
			print("\t- Invalid ACTION. Specify either \"extract\" or \"rate\" -")
		win.mainloop()
		return 

	def main(self): 
		"""
			Perform action corresponding to self.ACTION
		"""
		if self.ACTION == "extract":
			self.mask_extract_image()
		elif self.ACTION == "rate":
			self.rate_binarize_image()
		else:
			print("\t- Invalid action. Specify either \"extract\" or \"rate\" -")



