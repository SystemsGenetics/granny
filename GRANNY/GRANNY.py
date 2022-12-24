import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
import shutil
import skimage
from . import GRANNY_config as config
import re
import matplotlib.pyplot as plt
import cv2
from tkinter import * 
pd.options.mode.chained_assignment = None
tf.autograph.set_verbosity(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class GRANNY(object): 
	def __init__(self):
		# current directory
		self.ROOT_DIR = pathlib.Path(__file__).parent.resolve()

		# logs 
		self.MODEL_DIR = os.path.join(os.path.curdir, "logs")

		# new directory of the rotated input images
		self.NEW_DATA_DIR = "input_data" + os.sep

		# directory of the pretrained weights
		self.PRETRAINED_MODEL = os.path.join(self.ROOT_DIR, "mask_rcnn_balloon.h5")

		# initialize parameters
		self.VERBOSE = 0
		self.FILE_NAME = ""
		self.FOLDER_NAME = ""
		self.OLD_DATA_DIR = ""
		self.ACTION = ""

		# accepted file extensions 
		self.FILE_EXTENSION = (
			".JPG", ".JPG".lower(),
			".PNG", ".PNG".lower(),
			".JPEG", ".JPEG".lower(),
			".TIFF", ".TIFF".lower(),
		)

		# location where masked apple trays will be saved
		self.FULLMASK_DIR = "results" + os.sep + "full_masked_images" + os.sep

		# location where segmented/individual apples will be saved 
		self.SEGMENTED_DIR = "results" + os.sep + "segmented_images" + os.sep

		# location where apples with the scald removed will be saved
		self.BINARIZED_IMAGE = "results" + os.sep + "binarized_images" + os.sep


	def setParameters(self, action, fname, mode): 
		"""
			Setter method for action to perform in main()

			Args: 
				(str) action: action to run either mask_extract_image() or rate_binarize_image()
				(str) fname: data either a folder or a file
				(int) mode: specify 2 if fname is a folder (multiple images)

			Returns: 
				None	
		"""
		# set action to either "extract" or "rate"
		self.ACTION = action
	
		# set data folder/file 
		if self.ACTION == "extract": 
			self.OLD_DATA_DIR = fname
		elif self.ACTION == "rate": 
			self.FILE_NAME = fname
			self.FOLDER_NAME = fname

		# set mode for single/multiple-image processing
		if mode == None: 
			self.MODE = 1
		else: 
			self.MODE = mode
	

	def setVerbosity(self, verbose):
		""" 
			Setter method for configuration verbosity

			Args: 
				(int) verbose: specify 1 to display configuration summary

			Returns: 
				None
		"""
		self.VERBOSE = verbose


	def check_path(self, dir, reset = 0):
		"""
			Check and create a directory if not exist

			Args: 
				(str) dir: folder name
				(int) reset: specify 1 to reset if the directory exists

			Returns: 
				None
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
			
			Args: 
				(str) data_dir: the directory 

			Returns: 
				(list) folder_name: all folders inside data_dir
				(list) file_name: all files inside data_dir
		"""
		file_name = []
		folder_name = []

		# list all folders and files in data_dir
		for root, dirs, files in os.walk(data_dir):
			
			# append the files to the list
			for file in files:
				if file.endswith(self.FILE_EXTENSION):
					file_name.append(os.path.join(root, file))
			
			# append the folders to the list 
			for fold in dirs:
				folder_name.append(os.path.join(root,fold))
			if folder_name == []: 
				folder_name.append(os.path.join(root))
		
		return folder_name, file_name


	def clean_name(self, fname): 
		""" 
			Remove file extensions in file names 

			Args: 
				(str) fname: file names with extensions

			Returns: 
				(str) fname: file names without file extensions
		"""
		for ext in self.FILE_EXTENSION:
			fname = re.sub(ext, "", fname)
		return fname 


	def load_model(self, verbose = 1):
		""" 
			Load pretrained model, download if the model does not exist

			Args: 
				(int) verbose: specify 0 to turn off model display

			Returns: 
				(Keras model) model: model with the weights loaded
		"""
		# download the pretrained weights from GitHub if not exist
		if not os.path.exists(self.PRETRAINED_MODEL):
			config.MRCNN_utils.download_trained_weights(self.PRETRAINED_MODEL)
		
		# load the configurations for the model
		AppleConfig = config.AppleConfig()
		if verbose: 
			AppleConfig.display()
		
		# load model 
		model = config.MRCNN_model.MaskRCNN(mode = "inference", model_dir = self.MODEL_DIR, config = AppleConfig)

		# load pretrained weights to model
		model.load_weights(self.PRETRAINED_MODEL, by_name=True)
		return model

	
	def create_fullmask_image(self, model, im, fname = ""): 
		""" 
			Identify individual apples using the model 

			Args: 
				(Keras model) model: Mask-RCNN model
				(numpy.array) im: full image (tray of apples) to mask 
				(str) data_dir: directory to save the masked image 
				(str) fname: file (image) name

			Returns: 
				(numpy.array) mask: [height, width, num_instances]
				(numpy.array) box: [num_instance, (y1, x1, y2, x2, class_id)]
		"""
		# detect image's instances using the model 
		results = model.detect([im], verbose=0)
		r = results[0]

		# get the binary mask, box(coordinates), and confidence score from the result
		mask = r["masks"].astype(int)
		box  = r["rois"]
		score = r["scores"]
		class_names = ["BG", ""]

		# display the image with the masks, box, and scores
		config.MRCNN_visualize.display_instances(im, box, mask, r['class_ids'], 
								class_names, score)

		# save the figure
		plt.savefig(os.path.join(fname + ".png"), bbox_inches='tight')

		return mask, box
	

	def label_instances_helper(self, df): 
		""" 
			Helper function to sort apples using their center coordinates

			This sorting algorithm follows the numbering convention in 
			'01-input_data/GS-1-16_FilesForImageAnalysis/GS-1-16_ImageTutorial.pptx'. 
			In an increasing order, sort by y-center coordinates then sort by x-center coordinates. 

			Args:
			(DataFrame) df: panda DataFrame containing coordinate information 

			Returns:
			(list) df_list: sorted coordinates of apples/pears
		"""
		# sort df by y-center coordinates
		df = df.sort_values("ycenter", ascending= True, ignore_index = True)
		df.append(df.iloc[-1])

		# put the apples/pears into rows
		rows = 1
		count = 0
		for count in range(0, len(df)-1):
			df["rows"].iloc[count] = rows
			if not np.abs(df["ycenter"].iloc[count+1] - df["ycenter"].iloc[count]) < 300: 
				rows += 1
		df["rows"].iloc[-1] = rows

		# sort apple/pear in each row using their x-center coordinates
		# if the first row has 5 apples/pears
		df_list = []
		if len(df[df["rows"] == 1]) == 5:
			apple_id = 18
			for i in range(1,5):
				dfx = df[df["rows"] == i].sort_values("xcenter", ascending= False, inplace = False, ignore_index = True)
				for id in range(0, len(dfx)): 
					dfx["apple_id"].iloc[id] = apple_id
					apple_id -= 1
				df_list.append(dfx)

		# if the first row has 4 apples/pears
		else:
			apple_id = 1
			for i in range(1,5):
				dfx = df[df["rows"] == i].sort_values("xcenter", ascending= False, inplace = False, ignore_index = True)
				for id in range(0, len(dfx)): 
					dfx["apple_id"].iloc[id] = apple_id
					apple_id += 1
				df_list.append(dfx)
		
		return df_list 


	def sort_instances(self, box): 
		""" 
			Sort and identify apples 
			This sorting algorithm follows the numbering convention in 
			'01-input_data/GS-1-16_FilesForImageAnalysis/GS-1-16_ImageTutorial.pptx'

			Args:
			(numpy.array) box: [N, 4] where each row is y1, x1, y2, x2

			Returns:
			(list) apple_list: sorted coordinates of apples/pears 
		"""
		# convert to DataFrame
		df = pd.DataFrame(box)

		# label each column
		df.columns = ["y1", "x1", "y2", "x2"]

		# take first 18 rows (18 apples)
		df = df.iloc[0:18]

		# calculate centers for each apples
		df["ycenter"] = ((df["y1"]+df["y2"])/2).astype(int)
		df["xcenter"] = ((df["x1"]+df["x2"])/2).astype(int)

		# initialize columns
		df["rows"] = 0
		df["apple_id"] = 0
		df["nums"] = df.index

		# sort the DataFrame and return the list of instances
		apple_list = self.label_instances_helper(df)
		return apple_list
	

	def extract_image(self, df_list, mask, im, fname = ""): 
		""" 
			Extract individual image from masks created by Mask-RCNN 

			Args: 
				(list) df_list: sorted coordinates of apples/pears
				(numpy.array) mask: binary mask of individual apples
				(numpy.array) im: full image (tray of apples) to extract 
				(str) data_dir: directory to save the images
				(str) fname: file name

			Returns: 
				None
		"""
		# loop over 18 apples/pears
		for df in df_list: 
			
			# loop over the coordinates
			for i in range(0,len(df)):
				
				# convert to np.array 
				ar = np.array(df)

				# take the corresponsing mask 
				m = mask[:,:,ar[i][-1]]

				# initialize a blank array for the image
				new_im = np.zeros([ar[i][2]-ar[i][0],ar[i][3]-ar[i][1],3], dtype = np.uint8)

				# extract individual image from the coordinates
				for j in range(0,im.shape[2]):
					new_im[:,:,j] = im[ar[i][0]:ar[i][2],ar[i][1]:ar[i][3],j]*m[ar[i][0]:ar[i][2],ar[i][1]:ar[i][3]]

				# save the image
				plt.imsave(fname + "_" + str(ar[i][-2]) + ".png", new_im)

	
	def remove_purple(self, img): 
		"""
			Remove the surrounding purple from the individual apples using YCrCb color space. 
			This function helps remove the unwanted regions for more precise calculation of the scald area. 


			Args: 
			(numpy.array) img: RGB, individual apple image 

			Returns: 
			(numpy.array) new_img: RGB, individual apple image with no purple regions 
		"""
		# convert RGB to YCrCb
		new_img = img
		ycc_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

		# create binary matrix (ones and zeros)
		bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)

		# set max and min values for each channel 
		channel1Min = 0*bin
		channel1Max = 255*bin
		channel2Min = 0*bin
		channel2Max = 255*bin
		channel3Min = 0*bin
		channel3Max = 126*bin

		# create threshold matrices for each for each channel 
		threshold_1 = np.greater_equal(ycc_img[:,:,0], channel1Min) & np.less_equal(ycc_img[:,:,0], channel1Max)
		threshold_2 = np.greater_equal(ycc_img[:,:,1], channel2Min) & np.less_equal(ycc_img[:,:,1], channel2Max)
		threshold_3 = np.greater_equal(ycc_img[:,:,2], channel3Min) & np.less_equal(ycc_img[:,:,2], channel3Max)
		th123 = (threshold_1 & threshold_2 & threshold_3) 

		# create new image using threshold matrices 
		for i in range(3):
			new_img[:,:,i] = new_img[:,:,i] * th123
		return new_img 
	
	def smooth_binary_mask(self, bin_mask): 
		"""
			Smooth scald region with basic morphological operations. 
			By performing morphology, the binary mask will be smoothened to avoid discontinuity. 

			Args: 
				(numpy.array) bin_mask: binary mask (zeros & ones matrix) of the apples

			Returns: 
				(numpy.array) bin_mask: smoothed binary mask (zeros & ones matrix) of the apples
		"""
		bin_mask = np.uint8(bin_mask)

		# create a circular structuring element of size 20
		ksize = (20, 20)
		strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize = ksize)

		# using to structuring element to perform one close and one open operation on the binary mask 
		bin_mask = cv2.dilate(cv2.erode(bin_mask, kernel = strel, iterations = 1), kernel = strel, iterations = 1)
		bin_mask = cv2.erode(cv2.dilate(bin_mask, kernel = strel, iterations = 1), kernel = strel, iterations = 1)
		return bin_mask


	def segment_green(self, img): 
		"""
			Remove the scald region from the individual apple images. 
			Note that the stem could have potentially been removed during the process. 

			Args: 
				(numpy.array) img: RGB, individual apple image 

			Returns: 
				(numpy.array) new_img: RGB, individual apple image with the scald region removed
				(numpy.array) th123: binary mask 
		"""
		# convert from RGB to Lab color space
		new_img = img
		lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)

		# rescale all 3 channels to the 0-90 range
		for i in range(3):
			lab_img[:,:,i] = (lab_img[:,:,i] - np.amin(lab_img[:,:,i]))/(np.amax(lab_img[:,:,i])- np.amin(lab_img[:,:,i]))
			lab_img[:,:,i] *= 90
		
		# get channel 2 histogram for min and max values
		lab_img = lab_img.astype(np.uint8)
		hist, bin_edges = np.histogram(lab_img[:,:,1], bins = 90)

		# create binary matrix (ones and zeros)
		bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)

		# set max and min values for each channel 
		im_range = np.float32(90)
		channel1Min = 1*bin
		channel1Max = im_range*bin
		channel2Min = -1/2*im_range*bin
		channel2Max = (np.argmax(hist) - 2/9*im_range + 1)*bin
		channel3Min = 1*bin
		channel3Max = im_range*bin

		# create threshold matrices for each for each channel 
		threshold_1 = np.greater_equal(lab_img[:,:,0], channel1Min) & np.less_equal(lab_img[:,:,0], channel1Max)
		threshold_2 = np.greater_equal(lab_img[:,:,1], channel2Min) & np.less_equal(lab_img[:,:,1], channel2Max)
		threshold_3 = np.greater_equal(lab_img[:,:,2], channel3Min) & np.less_equal(lab_img[:,:,2], channel3Max)
		th123 = (threshold_1 & threshold_2 & threshold_3)

		# perform simple morphological operation to smooth the binary mask
		th123 = self.smooth_binary_mask(th123)

		# apply the binary mask on the image
		for i in range(3):
			new_img[:,:,i] = new_img[:,:,i] * th123
		return th123, new_img


	def score_image(self, img): 
		""" 
			Clean up individual image (remove purple area of the tray),
			and remove scald 

			Args: 
				(numpy.array) img: individual apple/pear image to rate

			Returns: 
				(numpy.array) nopurple_img: 
				(numpy.array) img: RGB, no scald image 
				(numpy.array) bw: binary mask (zeros & ones array)
		"""
		# Resize image to 800x800
		img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
		old_img = img

		# Remove surrounding purple
		img = self.remove_purple(img)
		nopurple_img = img
		
		# Image smoothing
		img = cv2.GaussianBlur(img, (3,3), sigmaX=0, sigmaY=0) 
		
		# Image binarization (Removing scald regions)
		bw, img = self.segment_green(img)

		return nopurple_img, img, bw

	def rotate_image(self, old_im_dir, new_im_dir = ""):
		"""
			Check and rotate image 90 degree if needed to get 4000 x 6000

			Args: 
				(str) old_im_dir: directory of the original image
				(str) new_im_dir: directory of the rotated image

			Returns: 
				(numpy.array) img: rotated image
		"""
		img = skimage.io.imread(old_im_dir)
		if img.shape[0]>img.shape[1]:
			img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		skimage.io.imsave(new_im_dir, img)
		return img

		
	def calculate_scald(self, bw, img):
		"""
			Calculate scald region by counting all non zeros area

			Args: 
				(numpy.array) bw: binarized image
				(numpy.array) img: original image to be used as ground truth

			Returns: 
				(float) fraction: the scald region, i.e. fraction of the original image that was removed
		"""
		# convert to uint8 
		img = np.uint8(img)

		# count non zeros of binarized image
		ground_area = 1/3*np.count_nonzero(img[:,:,0:2])
		
		# count non zeros of original image
		mask_area = 1/3*np.count_nonzero(bw[:,:,0:2])

		# calculate fraction 
		fraction = 0 
		fraction = 1 - mask_area/ground_area 
		if fraction < 0: 
			fraction = 0
		return fraction


	def rate_binarize_image(self): 
		""" 
			Main method performing Image Binarization, i.e. rate and remove scald, on individual apple images 
			The scores will be written to a .txt file

			Args: 
				None
			
			Returns: 
				None
		"""
		# create "results" directory to save the results
		self.check_path(self.BINARIZED_IMAGE)

		# single-image rating
		if self.MODE == 1:
			try:
				# read the image from file
				file_name = self.FILE_NAME
				img = skimage.io.imread(file_name)

				# remove the surroundings
				nopurple_img, binarized_image, bw = self.score_image(img)

				# calculate the scald region and save image 
				score = self.calculate_scald(binarized_image, nopurple_img)
				idx = -file_name[::-1].find(os.sep)
				file_name = file_name[idx:]
				skimage.io.imsave(os.path.join(self.BINARIZED_IMAGE, file_name), binarized_image)

				# save the scores to results/rating.txt
				with open("results" + os.sep +"rating.txt","w") as w:
					w.writelines(f"{self.clean_name(file_name)}:\t\t{score}")
					w.writelines("\n")
				print(f"\t- {self.clean_name(file_name)} rated. Check \"results/\" for output. - \n" )
			except FileNotFoundError:
				print(f"\t- Folder/File does not exist. -")
		
		# multi-images rating
		elif self.MODE == 2:
			try:
				# list all files and folders in the folder
				folders, files = self.list_all(self.FOLDER_NAME)

				# create "results" directory to save the results
				for folder in folders:
					self.check_path(folder.replace(self.FOLDER_NAME, self.BINARIZED_IMAGE))
				
				# remove scald and rate each apple
				scores = []
				for file_name in files: 

					# read the image from file
					img = skimage.io.imread(file_name)
					file_name = self.clean_name(file_name)

					# remove the surroundings
					nopurple_img, binarized_image, bw = self.score_image(img)

					# calculate the scald region and save image
					score = self.calculate_scald(binarized_image, nopurple_img)
					idx = -file_name[::-1].find(os.sep)
					file_name = file_name[idx:]
					scores.append(score)
					skimage.io.imsave(os.path.join(self.BINARIZED_IMAGE, file_name + ".png"), binarized_image)
				
				# save the scores to results/rating.txt
				with open("results" + os.sep + "ratings.txt","w") as w:
					for i, score in enumerate(scores): 
						w.writelines(f"{self.clean_name(files[i])}:\t\t{score}")
						w.writelines("\n")
						print(f"\t- {self.clean_name(file_name)} rated. Check \"results/\" for output. - \n" )
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

			Args: 
				None

			Returns: 
				None
		"""
		try: 
			# load model 
			model = self.load_model(verbose = self.VERBOSE)
			
			# list all folders and files 
			data_dirs, file_names = self.list_all(self.OLD_DATA_DIR)

			# check and create a new "results" directory to store the results 
			for data_dir in data_dirs: 
				self.check_path(data_dir.replace(self.OLD_DATA_DIR, self.FULLMASK_DIR))
				self.check_path(data_dir.replace(self.OLD_DATA_DIR, self.SEGMENTED_DIR))
				self.check_path(data_dir.replace(self.OLD_DATA_DIR, self.NEW_DATA_DIR))

			# pass each image to the model 
			for file_name in file_names:
				# get file name 
				idx = -file_name[::-1].find(os.sep)
				name = file_name[idx:]

				# print, for debugging purpose
				print(f"\t- Passing {name} into Mask R-CNN model. -")
				print(f"{name}") 

				# check and rotate the image to landscape (4000x6000)
				img = self.rotate_image(
					old_im_dir = file_name, 
					new_im_dir = file_name.replace(self.OLD_DATA_DIR, self.NEW_DATA_DIR), 
				)

				# remove file extension 
				file_name = self.clean_name(file_name)

				# use the MRCNN model, identify individual apples/pear on trays
				mask, box = self.create_fullmask_image(
					model = model, 
					im = img,
					fname = file_name.replace(self.OLD_DATA_DIR, self.FULLMASK_DIR)
				)

				# only take images that have at least 18 instances (18 apples/pears)
				if len(box) >= 18:
					df_list = self.sort_instances(box)
					self.extract_image(
						df_list = df_list,
						mask = mask,
						im = img,
						fname = file_name.replace(self.OLD_DATA_DIR, self.SEGMENTED_DIR)
					)
				
				# for debugging purpose
				print(f"\t- {name} extracted. Check \"results/\" for output. - \n" )
		except FileNotFoundError:
			print(f"\t- Folder/File Does Not Exist -")


	def main(self): 
		"""
			Perform action corresponding to self.ACTION
		"""
		# extract each individual instance from the full-tray image
		if self.ACTION == "extract":
			self.mask_extract_image()
		
		# (GS) rate each apple
		elif self.ACTION == "rate":
			self.rate_binarize_image()

		# not a valid action
		else:
			print("\t- Invalid action. Specify either \"extract\" or \"rate\" -")



