# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:27:48 2021

@author: ianni
"""

from pyseqlogo.pyseqlogo import draw_logo, setup_axis

from PIL import Image
import torch
import numpy as np
import os.path
import matplotlib.pyplot as plt

def NormalizeNumpy(image_3d):
    vmin = np.min(image_3d)
    image_2d = image_3d - vmin
    vmax = np.max(image_2d)
    return (image_2d / vmax)

def imshow(img):
    img = img     # unnormalize
    npimg = img.detach().numpy()
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg)
    plt.show()
    #    plt.savefig("imshowfig.png")


merge = 1
num = 0
rnge = int(70 / 2)

pred_labels = []
with open('attributions/label_predictions.txt', 'r') as output:
    for line in output:
        currentPlace = line[:-1]
        pred_labels.append(currentPlace)

dat = []
for i in range(100):
    dat.append('')
    genome_sequence = np.array(Image.open('attributions/genomes_saved/all/' + str(i) + '.png'))
    genome_sequence = np.mean(genome_sequence, axis=2)
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


for num in range(70, 100, 1):
    
    if os.path.isfile('attributions/human/' + str(num) + '.png'):
        label_ground_truth = 'human'
    elif os.path.isfile('attributions/goldfish/' + str(num) + '.png'):
        label_ground_truth = 'goldfish'
        
    if (label_ground_truth == pred_labels[num]):
        print(str(num) + ": " + str(dat[num]))
        print("GT: " + label_ground_truth + "    Pred: " + pred_labels[num])
        ## Display Vanilla Gradient ##
        img_attribution = np.array(Image.open('attributions/all/' + str(num) + '.png'))
        img_attribution = np.mean(img_attribution, axis=2)
#        img_attribution = NormalizeNumpy(img_attribution*img_attribution)
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
        
        ## Display Shift Smooth ##
        img_attribution_shsm = np.array(Image.open('attributions/all/' + str(num) + 'shsm.png'))
        img_attribution_shsm = np.mean(img_attribution_shsm, axis=2)
#        img_attribution_shsm = NormalizeNumpy(img_attribution_shsm*img_attribution_shsm)
        img_attribution_shsm = NormalizeNumpy(img_attribution_shsm)
        
        index = int(len(img_attribution_shsm[0])/2)
        attr_scores_shsm = []
        for p, letter in enumerate(dat[num][index - rnge : index + rnge]):
            attr_norm = NormalizeNumpy(img_attribution_shsm[0])
            font_size = (attr_norm[index - rnge + p])
            if letter == 'A':
                attr_scores_shsm.append([('A', font_size)])
            elif letter == 'T':
                attr_scores_shsm.append([('T', font_size)])
            elif letter == 'G':
                attr_scores_shsm.append([('G', font_size)])
            elif letter == 'C':
                attr_scores_shsm.append([('C', font_size)])
        
                     
        plt.rcParams['figure.dpi'] = 50
        fig, axarr = draw_logo(attr_scores_shsm, yaxis='probability')
        ax = axarr[0,0]
        setup_axis(axarr[0,0], axis='y', majorticks=1, minorticks=0.1)
        fig.tight_layout()
        print("Shift Smooth Attributions")
        