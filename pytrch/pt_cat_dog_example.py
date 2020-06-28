import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DATADIR = "/home/kishan/Desktop/kagglecatsanddogs_3367a/PetImages"
TESTING = "Testing"
CATEGORIES = ["Dog", "Cat"]
REBUILD_DATA = False

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    print("Running on the CPU")


class DogsVSCats:
    # 50x50 px image
    IMG_SIZE = 50
    training_data = []

    def create_and_store_training_data(self, data_save_dir_name):
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            # get the classification  (0 or a 1). 0=dog 1=cat
            class_num = CATEGORIES.index(category)

            for img in tqdm(os.listdir(path)):
                try:
                    # Convert image to array and grayscale
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    # Resize image to IMG_SIZExIMG_SIZE
                    img_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    # Save data as [features, label] : [image, hot_hot vector]
                    self.training_data.append([np.array(img_array), np.eye(2)[class_num]])
                    # plt.imshow(img_array, cmap='gray')  # graph it
                    # plt.show()  # display!
                except Exception as e:
                    print(e)
                # break
            # break

        print('Training dataset size:', len(self.training_data))
        # Shuffle training data so cat and dog training data is interleaved
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)

        return data_save_dir_name


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1: Convolutional (2D layer)
        # (input nodes, output nodes, window size (5x5))
        self.conv1 = nn.Conv2d(1, 32, 5)
        # layer 2: Convolutional (2D layer)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # layer 3: Convolutional (2D layer)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # create some random data to determine 2D->1D shape for flattening
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        # layer 4: Linear layer(s) (1D "fully connected" layer)
        self.fc1 = nn.Linear(self._to_linear, 512)  # flatten
        # layer 5: Output Linear layer (1D "fully connected" layer)
        # 2 output for 2 categories [cat, dog]
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # 1) Run rectified linear on the convolutional layers
        # 2) Max pooling of the result over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # Set shape for flattening
        if self._to_linear is None:
            # x.x * x.y * x.z
            self._to_linear = np.prod(x[0].shape)
        return x

    def forward(self, x):
        # Run through first 3 Convolutional 2D layers
        x = self.convs(x)
        # Flatten to run through 1D layers (convert 3D feature maps to 1D feature vectors)
        x = x.view(-1, self._to_linear)
        # Run through layer 4: 1D fully connected
        x = F.relu(self.fc1(x))
        # Run through layer 5: 1D fully connected output layer
        x = self.fc2(x)
        # softmax: outputs are a confidence score, adding up to 1.
        return F.softmax(x, dim=1)


# 1) Create and store the training data as numpy array (as needed)
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.create_and_store_training_data('cat_dog')

training_data = np.load("training_data.npy", allow_pickle=True)
print(f'Total training data size: {len(training_data)}')

# 2) Split training data into features and labels, as well as convert to a tensor:
print(f'Generating features/labels from training data...')
features = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
# Normalize features data to values between [0, 1]
features = features/255.0
labels = torch.Tensor([i[1] for i in training_data])

# plt.imshow(features[1], cmap="gray")
# plt.show()
# print(labels[1])

# 3) Reserve 10% of data for validation
VAL_PCT = 0.1
val_size = int(len(features) * VAL_PCT)
train_size = len(training_data) - val_size
print(f'Training data size: {train_size}')
print(f'Validation data size: {val_size}')

train_features = features[:train_size]
train_labels = labels[:train_size]

test_features = features[train_size:]
test_labels = labels[train_size:]

# 4) Train model
print("Training model...")
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
# use mean squared error for one_hot vectors
loss_function = nn.MSELoss()

BATCH_SIZE = 100
EPOCHS = 10

# Epochs: episodes/iterations over dataset
for epoch in range(EPOCHS):
    # From 0, to the len of train_features, stepping BATCH_SIZE at a time.
    for i in tqdm(range(0, len(train_features), BATCH_SIZE)):
        batch_features = train_features[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_labels = train_labels[i:i+BATCH_SIZE]

        # Move the dataset into the GPU
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        net.zero_grad()

        # List of network predictions
        outputs = net(batch_features)
        # calc and grab the loss value
        loss = loss_function(outputs, batch_labels)
        # apply the resulting loss backwards through the network's parameters
        loss.backward()
        # attempt to optimize weights to account for loss/gradients
        optimizer.step()

    print(f"Epoch: {epoch}. Loss: {loss}")

# 5) Test neural net against validation dataset
print("Validating model against test dataset...")
correct = 0
total = 0
for i in tqdm(range(0, len(test_features), BATCH_SIZE)):
    batch_features = test_features[i:i+BATCH_SIZE].view(-1, 1, 50, 50).to(device)
    batch_labels = test_labels[i:i+BATCH_SIZE].to(device)
    batch_out = net(batch_features)

    predicted_maxes = [torch.argmax(i) for i in batch_out]
    actual_maxes = [torch.argmax(i) for i in batch_labels]
    for predicted_class, real_class in zip(predicted_maxes, actual_maxes):
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))
