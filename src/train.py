import torch
from engine import train_one_epoch, evaluate
import utils
from Dataset import PneumoniaDataset
from FasterRCNN import *
import transforms as T

def get_transform(train):
	transforms = []
	# converts the image, a PIL image, into a PyTorch Tensor
	transforms.append(T.ToTensor())
#	if train:
#		# during training, randomly flip the training images
#		# and ground-truth for data augmentation
#		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)

def main(batch_size=1,):
	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# 2 box classes, dummy or pneumonia
	num_classes = 2
	# use our dataset and defined transformations
	dataset = PneumoniaDataset(dataRoot="../stage_2_train_images", infoFile="../stage_2_train_split_all.csv", transforms=get_transform(train=True))
	dataset_test = PneumoniaDataset(dataRoot="../stage_2_train_images", infoFile="../stage_2_val_split_all.csv", transforms=get_transform(train=False))

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, shuffle=True, num_workers=4,
		collate_fn=utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
		collate_fn=utils.collate_fn)

	# get the model using our helper function
	model = get_model_instance_segmentation_unmodified(num_classes)

	# move model to the right device
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]

	# optimizer = torch.optim.SGD(params, lr=0.005,
	#                             momentum=0.9, weight_decay=0.0005)
	optimizer = torch.optim.Adam(params, lr=0.001,
		betas=(0.9, 0.999),
		eps=1e-08, weight_decay=0,
		amsgrad=False)
	# and a learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
												   step_size=3,
												   gamma=0.1)

	# let's train it for 10 epochs
	num_epochs = 10

	for epoch in range(num_epochs):
		# train for one epoch, printing every 10 iterations
		train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
		# update the learning rate
		lr_scheduler.step()
		# evaluate on the test dataset
		evaluate(model, data_loader_test, device=device)

	print("That's it!")

if __name__ == '__main__':
	main()
