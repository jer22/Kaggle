import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def load_data() :

	train = pd.read_csv(r"../train.csv",dtype = np.float32)
	targets = train.label.values
	features = train.loc[:, train.columns != "label"].values / 255.

	return train, features, targets

def test_validate(model, error, optimizer, batch_size, nepochs, features, targets) :

	features_train, features_vali, targets_train, targets_vali = \
	train_test_split(features, targets, test_size = 0.2, random_state = 0)

	featuresTrain = torch.from_numpy(features_train).cuda()
	targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor).cuda()
	featuresVali = torch.from_numpy(features_vali).cuda()
	targetsVali = torch.from_numpy(targets_vali).type(torch.LongTensor).cuda()

	train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
	vali = torch.utils.data.TensorDataset(featuresVali,targetsVali)

	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
	vali_loader = torch.utils.data.DataLoader(vali, batch_size = batch_size, shuffle = False)
	
	count = 0
	loss_list = []
	iteration_list = []
	accuracy_list = []

	st = time.time()
	for epoch in range(nepochs):
		for i, (images, labels) in enumerate(train_loader):
			train = images.view(-1,1,28,28)
			train.requires_grad = True

			# Clear gradients
			optimizer.zero_grad()

			# Forward propagation
			outputs = model(train)

			# Calculate softmax and ross entropy loss
			loss = error(outputs, labels)

			# Calculating gradients
			loss.backward()

			# Update parameters
			optimizer.step()
			count += 1
			if count % 50 == 0:
				# Calculate Accuracy		 
				correct = 0
				total = 0
				# Iterate through test dataset
				for images, labels in vali_loader:

					inputs = images.view(-1,1,28,28)

					# Forward propagation
					outputs = model(inputs)

					# Get predictions from the maximum value
					predicted = torch.max(outputs, 1)[1]

					# Total number of labels
					total += len(labels)

					correct += (predicted == labels).sum()

				accuracy = float(correct) / total

				# store loss and iteration
				loss_list.append(loss)
				iteration_list.append(count)
				accuracy_list.append(accuracy)
				if count % 500 == 0:
					# Print Loss
					print('Iteration: {} Loss: {} Accuracy: {:.10f}'.format(count, loss, accuracy))

	ed = time.time()
	print('Total time: {:.10f}'.format(ed - st))

	# visualization loss 
	plt.plot(iteration_list,loss_list)
	plt.xlabel("Number of iteration")
	plt.ylabel("Loss")
	plt.title("CNN: Loss vs Number of iteration")
	plt.show()

	# visualization accuracy 
	plt.plot(iteration_list,accuracy_list,color = "red")
	plt.xlabel("Number of iteration")
	plt.ylabel("Accuracy")
	plt.title("CNN: Accuracy vs Number of iteration")
	plt.show()

def train_model(model, error, optimizer, batch_size, nepochs, features, targets) :
	featuresData = torch.from_numpy(features).cuda()
	targetsData = torch.from_numpy(targets).type(torch.LongTensor).cuda()

	data = torch.utils.data.TensorDataset(featuresData,targetsData)

	data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = False)

	for epoch in range(nepochs):
	    print('epoch:{}/{}'.format(epoch + 1, nepochs))
	    for i, (images, labels) in enumerate(data_loader):
	        train = images.view(-1,1,28,28)
	        train.requires_grad = True

	        # Clear gradients
	        optimizer.zero_grad()

	        # Forward propagation
	        outputs = model(train)

	        # Calculate softmax and ross entropy loss
	        loss = error(outputs, labels)

	        # Calculating gradients
	        loss.backward()

	        # Update parameters
	        optimizer.step()