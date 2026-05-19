# Week 2: Preprocessing + Baseline

## Goal

The goal of Week 2 was to prepare the Flickr8k dataset for image captioning and train a first baseline model.

## What was done

- Loaded Flickr8k images and captions.
- Cleaned captions by converting text to lowercase and removing punctuation.
- Added `<start>` and `<end>` tokens to each caption.
- Built a vocabulary using only the training captions.
- Split the dataset into train, validation, and test sets by image name.
- Extracted image features using a pretrained ResNet-50 CNN.
- Built and trained a simple CNN-LSTM baseline model.
- Evaluated the model using BLEU score.

## Preprocessing

The original captions were cleaned before training. The text was lowercased, punctuation was removed, and extra spaces were deleted. After cleaning, each caption received special tokens:

`<start> caption text <end>`

The vocabulary was created only from the training set to avoid data leakage. Rare words were replaced with `<unk>`.

## Dataset split

The dataset was split by unique image names, not by caption rows. This is important because each image has several captions. If we split by rows, the same image could appear in train and validation/test sets.

The split used:

- 80% train
- 10% validation
- 10% test

## Image feature extraction

A pretrained ResNet-50 model was used as the CNN encoder. The final classification layer was removed, and each image was converted into a 2048-dimensional feature vector.

These features were saved so the CNN does not need to process all images again every time.

## Baseline model

The baseline model used a CNN-LSTM architecture:

- ResNet-50 extracts image features.
- A linear layer projects image features into the embedding space.
- An embedding layer converts caption words into vectors.
- An LSTM decoder predicts the next word.
- A final linear layer outputs scores for all words in the vocabulary.

## Evaluation

The model was evaluated with BLEU score. BLEU compares the generated caption with the real captions for the same image.

The notebook calculates:

- BLEU-1
- BLEU-2
- BLEU-4

## Main result

A baseline CNN-LSTM model was trained and evaluated with BLEU score. This baseline is simple, but it proves that the full pipeline works: preprocessing, vocabulary building, feature extraction, model training, caption generation, and evaluation.

## Next steps

For Week 3, the model can be improved by training longer, tuning hyperparameters, using attention, improving decoding, or using a stronger encoder.
