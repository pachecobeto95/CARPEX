import os, config
import requests, sys, json, os
import numpy as np
from PIL import Image
import pandas as pd


"""
End Device Routine to test Edge device and cloud server
this file sends an image
"""


try:
	#This scipt sends an example image to the edge device.
	#df = pd.read_csv(config.PROCESSING_TIME_EDGE_PATH)
	#print(df)
	#sys.exit()
	filePath = os.path.join(config.DIR_NAME, "001.jpg")
	files = {"media": open("./001.jpg", "rb")}
	#files = {"media": Image.open("./001.jpg")}
	
	url = config.URL_EDGE + "/api/edge/recognition_cache"
	r = requests.post(config.URL_EDGE + "/api/edge/recognition_cache", files=files)
	#sendImg(url, filePath, "001.jpg") # this function also sends a image to edge device via post request


except Exception as e:
	print(e.args)