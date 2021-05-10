# ShiftSmoothedAttributions
### Ian Nielsen Systems Devices and Algorithms in Bioinformatics Final Project Spring 2021

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

2. From here click Send To -> File -> Create File.

![image](https://user-images.githubusercontent.com/36169018/117705297-e2e98700-b199-11eb-9b79-08450de4711a.png)

3. Once the file is finished downloading rename it to ``human_genome_c1.txt`` and place it in the ``Codon_Detection\raw_data`` folder.

4. Now that the data is downloaded move to project repository.

``cd ShiftSmoothedAttributions\Codon_Detection``

5. Run dataset formatter until you are satisfied with the size of the dataset.

``python generate_dataset.py``

6. Generate attribution maps using trained model.

``python getattributions_motif.py``

7. Follow the steps for the Quickstart for the Codon Detection Task to run experiments using newly generated attribution maps.
