import torch
import torchvision
from torchvision.transforms import v2

# If a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download and nromalization of data and split between train and test sets
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 16

trainsetfull = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#split in train and validation set
num_val = 5000
num_train = len(trainsetfull) - num_val # 45000

generator = torch.Generator().manual_seed(42)
valset, trainset = random_split(trainsetfull, [num_val, num_train], generator=generator)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# train set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Creating the CNN model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
      
net = Net()
net = net.to(device)
# setup of loss and optimization functions
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training for 50 epochs
for epoch in range(50):  # loop over the dataset multiple times

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
    net.eval()  # 1. Ativa o modo de avaliação (desativa dropout/batchnorm)

    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 2. Desativa gradientes (Economiza VRAM e acelera o hardware)
        for data in valloader:  # Dica: rode este loader com o dobro do batch size
            images, labels = data[0].to(device), data[1].to(device)
    
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcula as métricas finais da época
    epoch_val_loss = val_loss / len(valloader)
    epoch_val_acc = 100 * correct / total

    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        # Salva os pesos do melhor momento na GPU para usar depois
        torch.save(net.state_dict(), 'best_model_cifar10.pth')
        #print(f"--> Novo melhor modelo salvo com Val Acc: {bestr_val_acc:.2f}%")   
        patience = 0
    
    else:
        patience += 1

    # print statistics
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {epoch_train_loss:.3f}, accuracy: {epoch_train_acc:.3f}, val_loss: {epoch_val_loss:.3f}, val_accuracy: {epoch_val_acc:.3f}')
    running_loss = 0.0

    if patience == 5:
        break

print('Finished Training')

# Recalling the best model
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
        
        outputs = best_net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      
total_acuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {total_acuracy:.2f}%')

# Count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = best_net(images)
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
