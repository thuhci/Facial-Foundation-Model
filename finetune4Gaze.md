**This part is not yet complete**

# To finetune the model for Gaze 360 dataset

## Dataset preparation
1. Download the Gaze 360 dataset from [Gaze 360](https://gaze360.csail.mit.edu/) to the current folder.
2. Run the [preprocess/data_prepocessing_gaze360.py](./preprocess/data_processing_gaze360.py) script to normalize the dataset and labels.
3. Run the [preprocess/preprocess_gaze360.py](./preprocess/gaze360.py) to align the dataset labels, which will be generated to `saved/data/gaze360/`.

[**This part and the dataloader is checked, no bugs found**]

## Script
See `scripts/gaze360/` folder for the finetuning script.

[**Can run, but no good result.**]