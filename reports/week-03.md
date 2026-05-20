# Week 3: Improved Model

## Goal

The goal of Week 3 was to improve the Week 2 CNN-LSTM baseline for image captioning on the Flickr8k dataset. The improved model uses a stronger image encoder, dropout, and tuned training parameters.

## What was done

- Loaded the Flickr8k images and captions.
- Used the same cleaned captions from Week 2.
- Built the vocabulary from the training captions.
- Compared two encoder-decoder models:
  - VGG16 + LSTM as the baseline model.
  - ResNet50 + LSTM as the improved model.
- Added dropout to reduce overfitting.
- Tuned learning rate and batch size.
- Plotted training and validation loss.
- Evaluated both models using BLEU-1 and BLEU-2.
- Displayed good and bad generated caption examples.

## Model comparison

| Model | BLEU-1 | BLEU-2 | Notes |
|---|---:|---:|---|
| VGG16 + LSTM | XX | XX | baseline |
| ResNet50 + LSTM | XX | XX | improved with dropout and tuned hyperparameters |

Replace `XX` with the scores from `reports/week3_model_comparison.csv` after running the notebook.

## Training settings

Baseline model:
- Encoder: VGG16
- Decoder: LSTM
- Dropout: 0.0
- Learning rate: 0.001
- Batch size: 64

Improved model:
- Encoder: ResNet50
- Decoder: LSTM
- Dropout: 0.3
- Learning rate: 0.0005
- Batch size: 32

## Results discussion

The improved model is expected to perform better because ResNet50 produces stronger image features than the baseline encoder. Dropout also helps the decoder generalize better and reduces overfitting. The training and validation loss plot helps compare whether the improved model trains more smoothly than the baseline.

Good caption examples usually describe the main object or action correctly. Bad examples usually happen when the model predicts a common caption that does not match the specific image details.

## Files produced

- `reports/week3_training_history.csv`
- `reports/week3_model_comparison.csv`
- `reports/week3_good_caption_examples.csv`
- `reports/week3_bad_caption_examples.csv`

## Main result

The Week 3 improved CNN-LSTM model was trained and compared with the Week 2 baseline using BLEU-1 and BLEU-2 scores.
