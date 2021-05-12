"""
Created on Mon Apr 26 2021

@author: ianni
"""

from PIL import Image
import numpy as np
import os.path

genome = 'human'
merge = 1

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
    if not (os.path.isfile('data/Motif1/' + str(i) +'.png') | os.path.isfile('data/Motif2/' + str(i) +'.png') | os.path.isfile('data/MotifBoth/' + str(i) +'.png') | os.path.isfile('data/MotifNone/' + str(i) +'.png')):
        if not 'N' in seq:
            if (seq.__contains__('ATG') & seq.__contains__('TAA')):
                motif = 'MotifBoth'
            elif (seq.__contains__('ATG') & ~seq.__contains__('TAA')):
                motif = 'Motif1' # Start Codon
            elif (seq.__contains__('TAA') & ~seq.__contains__('ATG')):
                motif = 'Motif2' # Stop Codon
            else:
                motif = 'MotifNone'
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
                img.save("data_motif/" + motif + "/" + str(h) + ".png")
        else:
            k = k - 1
    else:
        if i % 1000 == 0:
            print('file ' + str(i) + ' already exists')







