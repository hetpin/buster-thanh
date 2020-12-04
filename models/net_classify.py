import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np


class NetClassify(nn.Module):
	"""A simple class to classify fake/real image
	input images: 1024*1024
	output: binary classification


	"""

	def __init__(self):
		super(NetClassify, self).__init__()
		self.no_classes = 2
		self.backbone_out_size = 1000
		# Tweak the resnet as feature extractor: input (1024x1024) -> 1000
		# feature vectors
		self.backbone = models.resnet18(pretrained=True)

		# print(self.backbone)

		# fc layers for classification
		self.fc = nn.Sequential(
			nn.Linear(
				in_features=self.backbone_out_size, out_features=512),
			# nn.BatchNorm1d(self.backbone_out_size),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=512, out_features=256),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=256, out_features=32),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=32, out_features=self.no_classes),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		"""forward path for the class

		Args:
			x (tensor): input image in batch style

		Returns:
			tensor: output
		"""
		x = self.backbone(x)
		x = self.fc(x)

		return x
