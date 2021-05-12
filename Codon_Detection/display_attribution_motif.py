"""
Created on Thu Apr 29 2021

@author: ianni
"""

from PIL import Image
import numpy as np
from PIL import ImageDraw, ImageFont 

def NormalizeNumpy(image_3d):
    vmin = np.min(image_3d)
    image_2d = image_3d - vmin
    vmax = np.max(image_2d)
    return (image_2d / vmax)

genome = 'human'
merge = 1
num = 4
rnge = int(70 / 2)

data = np.loadtxt("raw_data/" + genome + "_genome_c1.txt", delimiter="\n", dtype='str')


dat = []

for i in range(20):
    dat.append('')
    genome_sequence = np.array(Image.open('attributions_motif/genomes_saved/all/' + str(i) + '.png'))
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

img_attribution = np.array(Image.open('attributions_motif/all/' + str(num) + '.png'))
img_attribution = np.mean(img_attribution, axis=2)
img_attribution = NormalizeNumpy(img_attribution*img_attribution)

index = int(len(img_attribution[0])/2)
img = Image.new('RGB', (6000, 200), (255,255,255))
d = ImageDraw.Draw(img)
for p, letter in enumerate(dat[num][index - rnge : index + rnge]):
    font_size = int(img_attribution[0][index - rnge + p] * 200)
    if letter == 'A':
        fill_color = (0, 0, 0)
    elif letter == 'T':
        fill_color = (255, 0, 0)
    elif letter == 'G':
        fill_color = (0, 255, 0)
    elif letter == 'C':
        fill_color = (0, 0, 255)
    font = ImageFont.truetype("arial.ttf", font_size)
    d.text(((p * 75) - (font_size / 4) + 20, 20), dat[num][p], font=font, fill=fill_color)
img.show()

img_attribution = np.array(Image.open('attributions_motif/all/' + str(num) + 'shsm.png'))
img_attribution = np.mean(img_attribution, axis=2)
img_attribution = NormalizeNumpy(img_attribution*img_attribution)

img_shsm = Image.new('RGB', (6000, 200), (255,255,255))
d = ImageDraw.Draw(img_shsm)
for p, letter in enumerate(dat[num][index - rnge : index + rnge]):
    font_size = int(img_attribution[0][index - rnge + p] * 200)
    if letter == 'A':
        fill_color = (0, 0, 0)
    elif letter == 'T':
        fill_color = (255, 0, 0)
    elif letter == 'G':
        fill_color = (0, 255, 0)
    elif letter == 'C':
        fill_color = (0, 0, 255)
    font = ImageFont.truetype("arial.ttf", font_size)
    d.text(((p * 75) - (font_size / 4) + 20, 20), dat[num][p], font=font, fill=fill_color)
img_shsm.show()