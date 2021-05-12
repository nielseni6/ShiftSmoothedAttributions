"""
Created on Mon May  3 2021

@author: ianni
"""

import torch
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
import numpy as np
import torchvision.models as models
import os

def GetAttShiftSmooth(
  x_value, nshiftlr=1,
  magnitude=False):
    
    x_np = x_value.detach().cpu().numpy()
    
    total_gradients = torch.tensor(np.zeros_like(x_value.detach().cpu()))
    for i in range(nshiftlr*2 + 1):
        x_shifted = x_np
        x_shifted[0][0][0] = shift(x_np[0][0][0], i - nshiftlr)
        x_shifted[0][1][0] = shift(x_np[0][1][0], i - nshiftlr)
        x_shifted[0][2][0] = shift(x_np[0][2][0], i - nshiftlr)
        x_noise_tensor = torch.tensor(x_shifted, dtype = torch.float32)
        
        gradient = returnGradPred(x_noise_tensor.cuda())
        gradient[0][0][0][0] = torch.tensor(shift(gradient[0][0][0][0].detach().cpu().numpy(), nshiftlr - i))
        gradient[0][0][1][0] = torch.tensor(shift(gradient[0][0][1][0].detach().cpu().numpy(), nshiftlr - i))
        gradient[0][0][2][0] = torch.tensor(shift(gradient[0][0][2][0].detach().cpu().numpy(), nshiftlr - i))
        
        if magnitude:
            total_gradients += (gradient[0].cpu() * gradient[0].cpu())
        else:
            total_gradients += gradient[0].cpu()
    
    return total_gradients / (nshiftlr*2 + 1)

def shift(arr, num):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = arr[-num:]
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = arr[:-num]
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def NormalizeTensor(image_3d):
  vmin = torch.min(image_3d)
  image_2d = image_3d - vmin
  vmax = torch.max(image_2d)
  return (image_2d / vmax)

def returnGradPred(img):
    
    img.requires_grad_(True)
    pred = model(img)
    label = torch.tensor([int(torch.max(pred[0], 0)[1])])
    if (torch.cuda.is_available()):
        label = label.cuda()
    loss = criterion(pred, label)
    loss.backward()
    
    Sc_dx = img.grad
    
    return Sc_dx, pred

folders = ['attributions/all','attributions/human','attributions/goldfish',
           'attributions/genomes_saved/all','attributions/genomes_saved/human',
           'attributions/genomes_saved/goldfish']
for i, folder in enumerate(folders):
    for file in os.listdir(folder):
        if file.endswith('.png'):
            os.remove(os.path.join(folder,file))

transform = transforms.Compose([
                transforms.ToTensor(),
            ])

trainset = datasets.ImageFolder(root='data', transform=transform)

remainder = int(len(trainset) - (int(0.9 * len(trainset)) + int(0.1 * len(trainset))))
trainset, valset = torch.utils.data.random_split(trainset, [int(0.9 * len(trainset)), int(0.1 * len(trainset)) + remainder])

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

print("Training dataset size: ", len(trainset))
print("Validation dataset size: ", len(valset))

classifications = ["human", "goldfish"]

model = models.resnet18(pretrained=False)
## Modify ResNet number of outputs to match task at hand ##
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classifications))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if (torch.cuda.is_available()):
    model.cuda()

model.load_state_dict(torch.load("model.dth"))

# validation
model.eval()
total = 0
total_val_loss = 0
attributions, attr_shift_sm = [], []
images_saved, label_names = [], []
labels_saved = []
num_attr = 100
for itr, (image, label) in enumerate(val_dataloader):
    
    if (torch.cuda.is_available()):
        image = image.cuda()
        label = label.cuda()
        
    if itr < num_attr:
        attributions.append(returnGradPred(image))
        attr_shift_sm.append(GetAttShiftSmooth(image))
        images_saved.append(image)
        label_names.append(classifications[int(label)])
        labels_saved.append(int(label))
        
    pred = model(image)

    loss = criterion(pred, label)
    total_val_loss += loss.item()

    pred = torch.nn.functional.softmax(pred, dim=1)
    for i, p in enumerate(pred):
        if label[i] == torch.max(p.data, 0)[1]:
            total = total + 1

accuracy = total / len(valset)

total_val_loss = total_val_loss / (itr + 1)

print('\nVal Loss: {:.8f}, Val Accuracy: {:.8f}'.format(total_val_loss, accuracy))

for i in range(len(attributions)):
    images_saved[i] = torch.mean(images_saved[i], dim=1, keepdim=True)
    
    if labels_saved[i] == 0:
        label_n = 'human'
    elif labels_saved[i] == 1:
        label_n = 'goldfish'
    
    attr = NormalizeTensor(attributions[i][0].cpu())
    save_image(images_saved[i], "attributions/genomes_saved/" + label_n + "/" + str(i) + ".png")
    save_image(NormalizeTensor(attributions[i][0].cpu()), "attributions/" + label_n + "/" + str(i) + ".png")
    save_image(images_saved[i], "attributions/genomes_saved/all/" + str(i) + ".png")
    save_image(NormalizeTensor(attributions[i][0].cpu()), "attributions/all/" + str(i) + ".png")
    save_image(NormalizeTensor(attr_shift_sm[i][0].cpu()), "attributions/all/" + str(i) + "shsm.png")
    
with open("attributions/label_predictions.txt", "w") as output:
    for index in label_names:
        output.write('%s\n' % index)
    print('labels saved')
    



