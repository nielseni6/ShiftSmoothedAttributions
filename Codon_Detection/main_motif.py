# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import time

transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

trainset = datasets.ImageFolder(root='data_motif', transform=transform_train)
valset = datasets.ImageFolder(root='data_motif', transform=transform_test)

remainder = int(len(trainset) - (int(0.9 * len(trainset)) + int(0.1 * len(trainset))))
trainset, _ = torch.utils.data.random_split(trainset, [int(0.9 * len(trainset)), int(0.1 * len(trainset)) + remainder])
_, valset = torch.utils.data.random_split(valset, [int(0.9 * len(valset)), int(0.1 * len(valset)) + remainder])
#testset, valset = torch.utils.data.random_split(testset, [int(0.9 * len(testset)), int(0.1 * len(testset))])

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

classifications = ["ATG", "TAA", "ATG and TAA", "None"]

print("Training dataset size: ", len(trainset))
print("Validation dataset size: ", len(valset))
#print("Testing dataset size: ", len(testset))

model = models.resnet18(pretrained=False)
## Modify ResNet number of outputs to match task at hand ##
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classifications))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if (torch.cuda.is_available()):
    model.cuda()

no_epochs = 50
train_loss = list()
val_loss, val_acc = list(), list()
best_val_loss = 1
best_val_acc = 0
for epoch in range(no_epochs):
    start_time = time.time()
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    # training
    for itr, (image, label) in enumerate(train_dataloader):
        
#        image = torch.mean(image, dim=1, keepdim=True)
        
        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()
        
        optimizer.zero_grad()

        pred = model(image)
        pred = torch.nn.functional.softmax(pred, dim=1)

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()
        
        if itr % 25 == 0:
            print('Epoch: '+str(epoch)+' Iteration: '+str(itr))

    total_train_loss = total_train_loss / (itr + 1)
    train_loss.append(total_train_loss)

    # validation
    model.eval()
    total = 0
    for itr, (image, label) in enumerate(val_dataloader):
        
#        image = torch.mean(image, dim=1, keepdim=True)
        
        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        pred = model(image)

        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

    accuracy = total / len(valset)

    total_val_loss = total_val_loss / (itr + 1)
    val_loss.append(total_val_loss)
    val_acc.append(accuracy)

    print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))
    epoch_time = time.time() - start_time
    print("Time elapsed: " + str(epoch_time))
    
    torch.save(model.state_dict(), "model.dth")
    if total_val_loss <= best_val_loss:
        best_val_loss = total_val_loss
        print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, total_val_loss))
        torch.save(model.state_dict(), "model_best_loss.dth")
    if accuracy <= best_val_acc:
        best_val_acc = accuracy
        print("Saving the model state dictionary for Epoch: {} with Validation Acc: {:.8f}".format(epoch + 1, accuracy))
        torch.save(model.state_dict(), "model_best_acc.dth")

fig=plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, no_epochs+1), val_loss, label="Validation loss")
plt.xlabel('Loss')
plt.ylabel('Epochs')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show()

fig=plt.figure(figsize=(20, 10))
#plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, no_epochs+1), val_acc, label="Validation Accuracy")
plt.xlabel('Accuracy')
plt.ylabel('Epochs')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show()