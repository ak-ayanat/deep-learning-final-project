# Week 3: Improve Model

## Project
**Image Captioning with CNN and LSTM**

## Goal
The goal of Week 3 was to improve the baseline image captioning model by changing the CNN encoder and adding regularization. The improved model was compared with the Week 2 baseline using BLEU scores and validation loss.

## Dataset
The Flickr8k dataset was loaded from Kaggle.

- Total caption rows: **40,455**
- Unique images: **8,091**
- Train images: **6,472**
- Validation images: **809**
- Test images: **810**
- Vocabulary size: **2,655**

## Preprocessing
The captions were cleaned before training:

- converted text to lowercase
- removed punctuation
- added `<start>` and `<end>` tokens
- built vocabulary using the training captions only
- split data into train, validation, and test sets

## Models

### Baseline Model
The baseline model used:

- **VGG16** as CNN encoder
- LSTM decoder
- learning rate: **0.001**
- batch size: **64**
- dropout: **0.0**

### Improved Model
The improved model used:

- **ResNet50** as CNN encoder
- LSTM decoder
- dropout: **0.3**
- learning rate: **0.0005**
- batch size: **32**

## Training Results

| Model | Epoch | Train Loss | Validation Loss |
|---|---:|---:|---:|
| VGG16 + LSTM | 1 | 3.4713 | 2.9817 |
| VGG16 + LSTM | 2 | 2.7702 | 2.7187 |
| VGG16 + LSTM | 3 | 2.5070 | 2.6194 |
| VGG16 + LSTM | 4 | 2.3240 | 2.5701 |
| VGG16 + LSTM | 5 | 2.1740 | 2.5561 |
| ResNet50 + LSTM | 1 | 3.6482 | 3.0793 |
| ResNet50 + LSTM | 2 | 2.9599 | 2.7804 |
| ResNet50 + LSTM | 3 | 2.7084 | 2.6468 |
| ResNet50 + LSTM | 4 | 2.5505 | 2.5765 |
| ResNet50 + LSTM | 5 | 2.4396 | 2.5258 |

Both models improved during training. The ResNet50 model finished with a slightly lower validation loss than the VGG16 baseline.

## Model Comparison

| Model | BLEU-1 | BLEU-2 | Notes |
|---|---:|---:|---|
| VGG16 + LSTM | 0.5377 | 0.3539 | baseline |
| ResNet50 + LSTM | 0.5528 | 0.3740 | improved with dropout and tuned hyperparameters |

The improved ResNet50 + LSTM model achieved better BLEU-1 and BLEU-2 scores than the baseline. This shows that using a stronger encoder and dropout helped the model generate slightly better captions.

## Good Caption Examples

| Image | Prediction | Reference | BLEU-1 |
|---|---|---|---:|
| 1400424834_1c76e700c4.jpg | a woman in a black shirt and jeans is standing on a sidewalk | a lady dressed in shades of black waits on the sidewalk for a train | 0.9231 |
| 1598085252_f3219b6140.jpg | a small dog is running through a grassy field | a dog in a field | 0.8889 |
| 1807169176_7f5226bf5a.jpg | a group of people are standing around a campfire | a group of people sit around a blazing fire at night | 0.8889 |
| 1094462889_f9966dafa6.jpg | a brown dog is running through the snow | a brown dog plays in a deep pile of snow | 0.8750 |
| 1425069308_488e5fcf9d.jpg | a dog is playing with a white dog | a dog on two legs with its mouth opened toward a blue ball in the air | 0.8750 |

## Bad Caption Examples

| Image | Prediction | Reference | BLEU-1 |
|---|---|---|---:|
| 1797507760_384744fb34.jpg | two men in a blue shirt and a man in a blue shirt and a woman in a red shirt and a white shirt is | four shirtless men are hiking up a canyon | 0.0800 |
| 1191338263_a4fa073154.jpg | a man in a black shirt and a woman in a black shirt and a woman in a black shirt and a white hat is | a little old lady sitting next to an advertisement | 0.2000 |
| 1271210445_7f7ecf3791.jpg | a woman in a white shirt and a woman in a pink shirt is holding a baby | girls seated at table with a candle covered with lit candles | 0.2353 |
| 121971540_0a986ee176.jpg | a man in a blue shirt is standing on a rock | two men cleaning the outside windows of a yacht | 0.2490 |
| 1456393634_74022d9056.jpg | a girl in a pink bathing suit is running through the water | a boy doing a handstand on the beach | 0.2500 |

## Conclusion
In Week 3, the baseline model was improved by replacing VGG16 with ResNet50, adding dropout, and tuning the learning rate and batch size. The improved model had better BLEU scores:

- BLEU-1 improved from **0.5377** to **0.5528**
- BLEU-2 improved from **0.3539** to **0.3740**

The results show that ResNet50 + LSTM performs better than the VGG16 baseline, although the model still makes mistakes on complex scenes and sometimes repeats similar phrases.
