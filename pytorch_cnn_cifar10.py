import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import random_split

# If a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainsetfull = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# train and validation sets split
num_val = 5000
num_train = len(trainsetfull) - num_val  # 45000

generator = torch.Generator().manual_seed(42)
valset, trainset = random_split(trainsetfull, [num_val, num_train], generator=generator)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.drop(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        
        return x
      
net = Net()
net = net.to(device)
# setup of loss and optimization functions
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

best_val_acc = 0.0
patience = 0

for epoch in range(50):

    running_loss = 0.0
    train_total = 0
    train_correct = 0
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
    
        # acceleration with CPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * train_correct / train_total
        
        loss.backward()
        optimizer.step()
        
    #validation 
    net.eval() 

    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
    
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(valloader)
    epoch_val_acc = 100 * correct / total

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(net.state_dict(), 'best_model_cifar10.pth')
        patience = 0
    
    else:
        patience += 1

    # print statistics
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {epoch_train_loss:.3f}, accuracy: {epoch_train_acc:.3f}, val_loss: {epoch_val_loss:.3f}, val_accuracy: {epoch_val_acc:.3f}')
    running_loss = 0.0

    if patience == 5:
        break

print('Finished Training')

best_net = Net()
weights = torch.load('best_model_cifar10.pth', map_location=device, weights_only = True)
best_net.load_state_dict(weights)
best_net = best_net.to(device)

# evaluation of the model
correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      
total_acuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {total_acuracy:.2f}%')

# Count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
