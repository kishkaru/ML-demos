import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets


'''
Checklist:
1) That data is all numerical.
2) We've shuffled the data.
3) We've split the data into training and testing groups.
4) Is the data scaled. [0, 1]
5) Is the data balanced. (equal distribution of data features)
'''


# "Hello world" dataset: digits 0-9, 10 28x28px images
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

# Train using training set
# Small batch size helps reduce memorization:
# After each batch, the neural network does a back propagation for new, updated weights with hopes of decreasing loss.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

# Validate using validation set
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Input layer: "fully connected" layer
        # Automatically flattens 28x28px into 1x784px
        self.fc1 = nn.Linear(28*28, 64)
        # Hidden layer 1
        self.fc2 = nn.Linear(64, 64)
        # Hidden layer 2
        self.fc3 = nn.Linear(64, 64)
        # Output layer: 10 nodes (1 per each output prediction [0,9])
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # rectified linear (relu) activation function (keep data scaled between 0 and 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # softmax: outputs are a confidence score, adding up to 1.
        return F.log_softmax(x, dim=1)


net = Net()
# print(net)

# Calculates "how far off" our classifications are from reality
# use Cross Entropy for scalar classifications
# use mean squared error for one_hot vectors
loss_function = nn.CrossEntropyLoss()

# Adaptive Momentum (adam) optimizer
# lr = learning rate: Dictates the magnitude of changes that the optimizer can make at a time
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Epochs: episodes/iterations over dataset
for epoch in range(3):
    # Iterate over each batch (of 10 data elements)
    for data in trainset:  # `data` is a batch of data
        # data[0] = features
        # data[1] = labels
        features, labels = data
        net.zero_grad()  # set gradients to 0 before calc loss. Don't re-optimize for previous gradients that we already optimized for
        # List of network predictions
        output = net(features.view(-1, 28*28))  # pass in the reshaped batch
        loss = F.nll_loss(output, labels)  # calc and grab the loss value
        loss.backward()  # apply the resulting loss backwards through the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    print(loss)

# Test neural net against validation dataset
correct = 0
total = 0
with torch.no_grad():
    # Iterate over each batch (of 10 data elements)
    for data in testset:
        features, labels = data
        # List of network predictions
        output = net(features.view(-1, 28*28))

        for i, e in enumerate(output):
            if torch.argmax(e) == labels[i]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

