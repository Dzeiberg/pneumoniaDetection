import pandas as pd
import numpy as np
import torch
from PIL import Image
import os
import pydicom
from ast import literal_eval


class PneumoniaDataset(object):
	def __init__(self, dataRoot, infoFile, transforms):
		self.info = pd.read_csv(infoFile)
		self.dataRoot = dataRoot
		self.transforms = transforms

	def __getitem__(self, idx):
		row = self.info.iloc[idx]
		dcmPath = os.path.join(self.dataRoot,row["patientId"]) + ".dcm"
		dcm = pydicom.dcmread(dcmPath)
		pixels = dcm.pixel_array
		img = Image.fromarray(pixels)
		# Populate necessary fields
		image_id = idx
		# ADD dummy box to issue for control patients
		boxes = [[0,0,50,50]]
		labels = [0]
		area = [250]
		iscrowd = [False]
		target = None
		if "nan" not in row["x"]:
			xList = literal_eval(row["x"])
			yList = literal_eval(row["y"])
			widthList = literal_eval(row["width"])
			heightList = literal_eval(row["height"])
			label = row["Target"]
			for boxNum in range(len(xList)):
				x,y,w,h = xList[boxNum], yList[boxNum], widthList[boxNum], heightList[boxNum]
				boxes.append([x,y,x+w,y+h])
				labels.append(label)
				area.append(w * h)
				iscrowd.append(False)
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.as_tensor(labels, dtype=torch.int64)
		area = torch.as_tensor(area, dtype=torch.float32)
		iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
		image_id = torch.tensor([image_id])
		target = {"boxes":boxes,
				"labels":labels,
				"image_id":image_id,
				"area":area,
				"iscrowd":iscrowd}
		if self.transforms is not None:
			img, target = self.transforms(img, target)
		return img, target

	def __len__(self):
		return self.info.shape[0]


