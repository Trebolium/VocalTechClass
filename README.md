# Vocal Technique Classifier

## About this Repo

This repository contains a CRNN neueral network with an appended attention mechanism, designed to detect features that assist in classifying singing techniques in recordings singing clips. Our published paper [Zero-shot Singing Voice Conversion](https://cmmr2021.github.io/proceedings/pdffiles/cmmr2021_26.pdf) provides an overview of the architecture. This network allows us to generate singing-technique descriminative embeddings that can be used for downstream tasks where disentangled singing-technique information is required. 

## Directories

The provided dataset `example_ds` is used by default, and represents a minimal example of the required directory tree structure for a dataset, as well as the features themselves (mel-spectrograms). It is advisable to use the [VocalSet dataset](https://zenodo.org/record/1203819#.YiszFRDP0RY) as this is one of the few datasets that provide singing technique information.

The directory 'results' is created automatically if it does not exists. When `--file_name` is specified to be anything other than 'DefaultName', a new directory with that name is stored in the 'results' directory, where affiliated model-specific data will be stored.

Using the `--load_ckpt` attribute allows you to start training with saved weights from a previous session.

## Installation

To run the network, please ensure you have a virtual environment that has the following and their dependancies installed:

* torch==1.8.1
* ibrosa==0.8.0
* soundfile==0.10.3.post1,
* pydub==0.24.1,
* PyYAML==5.3.1,
* scipy==1.5.4,
* numpy==1.19.4,
* matplotlib==3.3.3,
* tensorboard==2.4.1,
* tensorboard-plugin-wit==1.8.0

## Training


Running `python main.py` will train the model using the toy example dataset. This will not generate strong accuracy values and not be fully trained due to the size of the dataset.

Using the full VocalSet dataset (trimmed to include the 5 techniques we're interested in), we replicated the basic CNN network described in VocalSet's associated paper, and scored 57%. However with this customly designed network, we can achieve up to 86% accuracy. We hypothesize that higher scores could not be achieved due to categorical leakage among classes, the size of the dataset, and several performance consistency issues detailed in our paper.

If you wish to use Wilkins' proposed network, use `--model=wilkins` after `python main.py`. You will also need to ensure that `--data_dir` points to a directory that contains a dataset of audio files. Do explore the other argument attributes to consider using non-default hyperparameters to experiment.