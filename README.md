# ShiftSmoothedAttributions
#### Ian Nielsen Systems Devices and Algorithms in Bioinformatics Final Project Spring 2021

## Install

Clone repo.

``git clone https://github.com/nielseni6/ShiftSmoothedAttributions.git``

## Getting Started

This documentation is split into two parts, **Quickstart** and **Training**. If you simply wish to recreate the results using the precalculated attribution maps then you will want to begin with **Quickstart**. If you would like to recreate all experiments from scratch, including formatting the dataset, training the model and generating attribution maps then you will want to begin from **Training**.

This repo uses the pyseqlogo package to display attributions. If you are getting errors with this package please download the pyseqlogo files from their github (https://github.com/saketkc/pyseqlogo) and place a copy of the pyseqlogo folder into the ``ShiftSmoothedAttributions\Codon_Detection`` and ``ShiftSmoothedAttributions\Human_Goldfish_Classification`` folders.

## Quickstart:

### Codon Detection Task

Move to project repository.

``cd ShiftSmoothedAttributions\Codon_Detection``

To recreate Experiment 1 (Shifting Invariance) run the shift experiment file.

``python shift_experiment.py``

To recreate Experiment 2 (Are the Areas of Interest Being Highlighted?) Run the display logo file to display the attribution maps given in this repo.

``python disp_attr_motif_logo.py``


### Human/Goldfish Classification Task

Move to project repository.

``cd ShiftSmoothedAttributions\Human_Goldfish_Classification``

To recreate Experiment 1 (Shifting Invariance) run the shift experiment file.

``python shift_experiment.py``

To recreate Experiment 2 (Are the Areas of Interest Being Highlighted?) Run the display logo file to display the attribution maps given in this repo. **Note: this experiment will not be able to validate the method the same as the codon detection task since the important features are not known for human/goldfish classification.**

``python disp_attr_logo.py``


## Train Model:
**If you would like to train the model yourself follow these steps**

### Codon Detection Task

1. Go to https://www.ncbi.nlm.nih.gov/nuccore/CM000663.2 and click on FASTA.

![image](https://user-images.githubusercontent.com/36169018/117705020-86866780-b199-11eb-8538-29cb9df8de3c.png)

2. From here click Send To -> File -> Create File, then Save File.

![image](https://user-images.githubusercontent.com/36169018/117705297-e2e98700-b199-11eb-9b79-08450de4711a.png)

3. Once the file is finished downloading rename it to ``human_genome_c1.txt`` and place it in the ``Codon_Detection\raw_data`` folder.

4. Now that the data is downloaded move to project repository.

``cd ShiftSmoothedAttributions\Codon_Detection``

5. Run dataset formatter until you are satisfied with the size of the dataset.

``python generate_dataset.py``

6. Generate attribution maps using trained model.

``python getattributions_motif.py``

7. Follow the steps for the **Quickstart** for the **Codon Detection Task** to run experiments using newly generated attribution maps.

### Human/Goldfish Classification Task

1. Go to https://www.ncbi.nlm.nih.gov/nuccore/CM000663.2 and click on FASTA.

![image](https://user-images.githubusercontent.com/36169018/117705020-86866780-b199-11eb-8538-29cb9df8de3c.png)

2. From here click Send To -> File -> Create File, then Save File.

![image](https://user-images.githubusercontent.com/36169018/117705297-e2e98700-b199-11eb-9b79-08450de4711a.png)

3. Once the file is finished downloading rename it to ``human_genome_c1.txt`` and place it in the ``Human_Goldfish_Classification\raw_data`` folder.

**Steps 4 through 6 are a repeat of steps 1 through 3 except that we are downloading the goldfish genome this time rather than human.**
4. Go to https://www.ncbi.nlm.nih.gov/nuccore/CM010432.1 and click on FASTA.

![image](https://user-images.githubusercontent.com/36169018/117706664-969f4680-b19b-11eb-8cf7-6e62d3b62b69.png)

5. From here click Send To -> File -> Create File, then Save File.

![image](https://user-images.githubusercontent.com/36169018/117707281-61472880-b19c-11eb-9b12-a08c15f87b3e.png)

6. Once the file is finished downloading rename it to ``goldfish_genome_c1.txt`` and place it in the ``Human_Goldfish_Classification\raw_data`` folder.

7. Now that the data is downloaded move to project repository.

``cd ShiftSmoothedAttributions\Human_Goldfish_Classification``

8. Run dataset formatter until you are satisfied with the size of the dataset.

``python generate_dataset.py``

9. Generate attribution maps using trained model.

``python getattributions.py``

10. Follow the steps for the **Quickstart** for the **Human/Goldfish Classification Task** to run experiments using newly generated attribution maps.
