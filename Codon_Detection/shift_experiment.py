"""
Created on Mon May  3 2021

@author: ianni
"""

from pyseqlogo.pyseqlogo import draw_logo, setup_axis

import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
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

def NormalizeNumpy(image_3d):
    vmin = np.min(image_3d)
    image_2d = image_3d - vmin
    vmax = np.max(image_2d)
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

rnge = int(70 / 2)

transform = transforms.Compose([
                transforms.ToTensor(),
            ])

trainset = datasets.ImageFolder(root='data_motif', transform=transform)

remainder = int(len(trainset) - (int(0.9 * len(trainset)) + int(0.1 * len(trainset))))
trainset, valset = torch.utils.data.random_split(trainset, [int(0.9 * len(trainset)), int(0.1 * len(trainset)) + remainder])

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

print("Training dataset size: ", len(trainset))
print("Validation dataset size: ", len(valset))

classifications = ["ATG", "TAA", "ATG and TAA", "None"]

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
attr_shift_sg = []
images_saved, label_names = [], []
labels_saved = []
num_attr = 100
for itr, (image, label) in enumerate(val_dataloader):
    
    if (torch.cuda.is_available()):
        image = image.cuda()
        label = label.cuda()
        
    if itr < num_attr:
        for i in range(5):
            image_shifted = image.detach().cpu().numpy()
            image_shifted[0][0][0] = shift(image[0][0][0].detach().cpu(), i - 2)
            image_shifted[0][1][0] = shift(image[0][1][0].detach().cpu(), i - 2)
            image_shifted[0][2][0] = shift(image[0][2][0].detach().cpu(), i - 2)
            attributions.append(returnGradPred(torch.tensor(image_shifted).cuda()))
            attr_shift_sm.append(GetAttShiftSmooth(torch.tensor(image_shifted).cuda()))
            images_saved.append(image_shifted)
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

dat = []
for i in range(10):
    dat.append('')
    genome_sequence = images_saved[i][0]
    genome_sequence = np.mean(genome_sequence, axis=0) * 255
    for j, letter in enumerate(genome_sequence[0]):
        if genome_sequence[0][j] <= 63:
            dat[i] = str(dat[i]) + 'A'
        elif genome_sequence[0][j] <= 127:
            dat[i] = str(dat[i]) + 'T'
        elif genome_sequence[0][j] <= 191:
            dat[i] = str(dat[i]) + 'G'
        elif genome_sequence[0][j] <= 255:
            dat[i] = str(dat[i]) + 'C'
        else:
            dat[i] = str(dat[i]) + 'N'

for num in range(10):
    
    if os.path.isfile('attributions_motif/Motif1/' + str(num) + '.png'):
        label_ground_truth = 'ATG'
    elif os.path.isfile('attributions_motif/Motif2/' + str(num) + '.png'):
        label_ground_truth = 'TAA'
    elif os.path.isfile('attributions_motif/MotifBoth/' + str(num) + '.png'):
        label_ground_truth = 'ATG and TAA'
    elif os.path.isfile('attributions_motif/MotifNone/' + str(num) + '.png'):
        label_ground_truth = 'None'
        
    print(str(num) + ": " + str(dat[num]))
    print("GT: " + label_ground_truth + "    Pred: " + label_names[num])
    
    ## Display Vanilla Gradient ##
    img_attribution = attributions[num][0][0].detach().cpu().numpy()
    img_attribution = np.mean(img_attribution, axis=0)
#    img_attribution = NormalizeNumpy(img_attribution*img_attribution) # Uncomment to use squared attribution maps
    img_attribution = NormalizeNumpy(img_attribution)
    
    index = int(len(img_attribution[0])/2)
    attr_scores_vanilla = []
    for p, letter in enumerate(dat[num][index - rnge : index + rnge]):
        attr_norm = NormalizeNumpy(img_attribution[0])
        font_size = (attr_norm[index - rnge + p])
        if letter == 'A':
            attr_scores_vanilla.append([('A', font_size)])
        elif letter == 'T':
            attr_scores_vanilla.append([('T', font_size)]) 
        elif letter == 'G':
            attr_scores_vanilla.append([('G', font_size)]) 
        elif letter == 'C':
            attr_scores_vanilla.append([('C', font_size)]) 
    
                 
    plt.rcParams['figure.dpi'] = 50
    fig, axarr = draw_logo(attr_scores_vanilla, yaxis='probability')
    ax = axarr[0,0]
    setup_axis(axarr[0,0], axis='y', majorticks=1, minorticks=0.1)
    fig.tight_layout()
    print("Vanilla Attributions")
    
for num in range(10):
    
    if os.path.isfile('attributions_motif/Motif1/' + str(num) + '.png'):
        label_ground_truth = 'ATG'
    elif os.path.isfile('attributions_motif/Motif2/' + str(num) + '.png'):
        label_ground_truth = 'TAA'
    elif os.path.isfile('attributions_motif/MotifBoth/' + str(num) + '.png'):
        label_ground_truth = 'ATG and TAA'
    elif os.path.isfile('attributions_motif/MotifNone/' + str(num) + '.png'):
        label_ground_truth = 'None'
        
    print(str(num) + ": " + str(dat[num]))
    print("GT: " + label_ground_truth + "    Pred: " + label_names[num])
    
    ## Display Vanilla Gradient ##
    img_attribution = attr_shift_sm[num][0][0].detach().cpu().numpy()
#    img_attribution = NormalizeNumpy(img_attribution*img_attribution) # Uncomment to use squared attribution maps
    img_attribution = NormalizeNumpy(img_attribution)
    
    index = int(len(img_attribution[0])/2)
    attr_scores_vanilla = []
    for p, letter in enumerate(dat[num][index - rnge : index + rnge]):
        attr_norm = NormalizeNumpy(img_attribution[0])
        font_size = (attr_norm[index - rnge + p])
        if letter == 'A':
            attr_scores_vanilla.append([('A', font_size)])
        elif letter == 'T':
            attr_scores_vanilla.append([('T', font_size)]) 
        elif letter == 'G':
            attr_scores_vanilla.append([('G', font_size)]) 
        elif letter == 'C':
            attr_scores_vanilla.append([('C', font_size)]) 
    
                 
    plt.rcParams['figure.dpi'] = 50
    fig, axarr = draw_logo(attr_scores_vanilla, yaxis='probability')
    ax = axarr[0,0]
    setup_axis(axarr[0,0], axis='y', majorticks=1, minorticks=0.1)
    fig.tight_layout()
    print("Shift Smooth Attributions")
