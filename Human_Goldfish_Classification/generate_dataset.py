"""
Created on Mon Apr 26 2021

@author: ianni
"""

from PIL import Image
import numpy as np
import os.path

## Change between goldfish and human to format data for each ##
#genome = 'goldfish' 
genome = 'human'
merge = 100

data = np.loadtxt("raw_data/" + genome + "_genome_c1.txt", delimiter="\n", dtype='str')

dat = []
for i, seq in enumerate(data):
    if i % merge == 0:
        dat.append(seq)
    else:
        dat[int(i/merge)] = dat[int(i/merge)] + seq
k = 0
im = np.zeros(len(dat[0]))
for i, seq in enumerate(dat):
    if not os.path.isfile("data/" + genome + '/' + str(i) +'.png'):
        if not 'N' in seq:
            for j, letter in enumerate([char for char in seq]):
                
                if letter == 'A':
                    im[j] = 63
                elif letter == 'T':
                    im[j] = 127
                elif letter == 'G':
                    im[j] = 191
                elif letter == 'C':
                    im[j] = 255
                else:
                    im[j] = 0
                h = k + i
                img = Image.fromarray(np.array([im],dtype=np.uint8))
                img.save("data/" + genome + "/" + str(h) + ".png")
        else:
            k = k - 1
    else:
        if i % 1000 == 0:
            print('file ' + str(i) + ' already exists')