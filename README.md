# Image Captioning with CNN and LSTM

## Project Overview
This project implements an Encoder-Decoder neural network architecture to automatically generate text descriptions for images. 
It uses a pretrained CNN (like ResNet or VGG) as the encoder to extract image features, and an LSTM as the decoder to generate captions.

## Dataset
We use the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k). 
Please see `data/README.md` for download instructions.

## Project Structure
- `data/` - dataset instructions (raw data is ignored).
- `notebooks/` - Jupyter notebooks for Exploratory Data Analysis (EDA) and experiments.
- `src/` - python scripts for models and training.
- `reports/` - weekly progress updates.
