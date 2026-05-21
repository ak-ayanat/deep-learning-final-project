# Flickr8k Image Captioning with ResNet50 + LSTM

## Project Overview

This project focuses on automatic image caption generation. The main goal of the project is to build a deep learning system that can take an image as input and generate a short natural language caption for it.

The project was completed during Week 1–Week 4. Each week focused on a different stage of the deep learning pipeline: dataset analysis, preprocessing, baseline model, improved model, evaluation, saving artifacts, and application development.

---

## Problem Statement

Images contain visual information, but computers cannot directly describe them in human language without a trained model. Image captioning solves this problem by combining computer vision and natural language processing.

This project solves the task by using a CNN encoder to understand the image and an LSTM decoder to generate a text caption word by word.

The final result is a trained CNN-LSTM image captioning model and a Streamlit mini application where the user can upload an image and receive a generated caption.

---

## Dataset

The dataset used in this project is the Flickr8k dataset.

The dataset contains images and human-written captions. Each image usually has 5 different captions written by people.

Dataset summary:

| Item | Value |
|---|---:|
| Total caption rows | 40,455 |
| Unique images | 8,091 |
| Captions per image | Usually 5 |
| Image folder | `/content/flickr8k/Images` |
| Caption file | `/content/flickr8k/captions.txt` |

This is an image captioning task. It means that the model should learn the relationship between an image and a sequence of words.

For example, if the input image contains a person, the model should generate a caption like:

```text
a man in a black shirt is standing
```

---

## Technologies Used

This project was implemented using:

- Python
- PyTorch
- TorchVision
- Pandas
- NumPy
- Matplotlib
- PIL
- NLTK
- Scikit-learn
- Streamlit
- Google Colab
- GitHub

---

# Week 1: Dataset Loading and Exploratory Data Analysis

## Goal of Week 1

The main goal of Week 1 was to load the Flickr8k dataset, understand its structure, and analyze the captions before training the model.

## Work Completed

During Week 1, the following tasks were completed:

- Loaded the Flickr8k dataset from Kaggle
- Loaded the captions file
- Checked dataset structure
- Checked image names and captions
- Counted caption lengths
- Plotted caption length distribution
- Displayed a random image with its captions

## Dataset Output

The dataset contains two main parts:

| Column | Description |
|---|---|
| image | Image file name |
| caption | Human-written caption for the image |

The exploratory analysis showed that the dataset contains image-caption pairs. One image can appear several times because each image has multiple captions.

## Week 1 Result

From Week 1 analysis, we understood the structure of the Flickr8k dataset. The dataset is suitable for image captioning because it contains real images and several captions for each image.

Caption length analysis was useful because it helped decide a reasonable maximum caption length for training. Random image visualization helped check that captions correctly match the images.

---

# Week 2: Preprocessing and Baseline Model

## Goal of Week 2

The goal of Week 2 was to clean the captions, build a vocabulary, split the data, extract image features, and train the first baseline CNN-LSTM model.

## Work Completed

During Week 2, the following tasks were completed:

- Cleaned captions
- Converted captions to lowercase
- Removed punctuation
- Removed extra spaces
- Added `<start>` and `<end>` tokens
- Built vocabulary using training captions
- Removed very rare words using minimum frequency
- Converted words into numerical indexes
- Split images into train, validation, and test sets
- Extracted image features using pretrained VGG16
- Built and trained a baseline LSTM decoder

## Text Preprocessing

Text preprocessing was important because raw captions may contain capital letters, punctuation, and unnecessary spaces. Cleaning captions makes the vocabulary smaller and easier for the model to learn.

Example:

```text
Original caption:
A man in a black shirt is standing near a window.

Cleaned caption:
<start> a man in a black shirt is standing near a window <end>
```

## Data Split

The dataset was split by unique image names, not only by caption rows. This is important because captions from the same image should not appear in both training and testing sets.

| Split | Images | Caption rows | Purpose |
|---|---:|---:|---|
| Train | 6,472 | 32,360 | Used to train the model |
| Validation | 809 | 4,045 | Used to check validation loss and tune settings |
| Test | 810 | 4,050 | Used for final evaluation and examples |

## Preprocessing Output

| Preprocessing step | Output / reason |
|---|---|
| Lowercase captions | Reduces duplicate word forms such as `Dog` and `dog` |
| Remove punctuation | Simplifies vocabulary |
| Add `<start>` token | Shows where caption generation begins |
| Add `<end>` token | Shows where caption generation stops |
| Minimum word frequency = 5 | Removes very rare words |
| Padding index ignored in loss | Prevents padded positions from affecting training |

## Baseline Model

The baseline model used in Week 2 was:

```text
VGG16 feature extractor + LSTM decoder
```

VGG16 was used as a pretrained CNN encoder. It extracted visual features from each image. Then the LSTM decoder used these features to generate captions.

## Week 2 Result

The Week 2 baseline created the first working image captioning model. It showed that CNN features can be connected with an LSTM decoder to generate captions.

The baseline was useful because it gave a comparison point for the improved model in Week 3.

---

# Week 3: Improved ResNet50 + LSTM Model

## Goal of Week 3

The goal of Week 3 was to improve the baseline model by using a stronger CNN encoder and better training settings.

## Work Completed

During Week 3, the following tasks were completed:

- Compared VGG16 encoder with ResNet50 encoder
- Used pretrained ResNet50 for image feature extraction
- Added dropout to reduce overfitting
- Changed learning rate
- Trained the improved CNN-LSTM model
- Compared train loss and validation loss
- Evaluated generated captions using BLEU scores
- Saved training history and comparison outputs
- Displayed good and bad caption examples

## Improved Model

The improved model used:

```text
ResNet50 feature extractor + LSTM decoder
```

ResNet50 was selected because it is a deeper and stronger CNN than VGG16. It produces a 2048-dimensional image feature vector, which is then passed to the LSTM decoder.

## Model Settings

| Model | Feature dimension | Learning rate | Batch size | Dropout | Epochs |
|---|---:|---:|---:|---:|---:|
| VGG16 + LSTM baseline | 4096 | 0.001 | 64 | 0.0 | 5 |
| ResNet50 + LSTM improved | 2048 | 0.0005 | 64 | 0.3 | 5 |

## Model Architecture

```text
Input Image
↓
Pretrained ResNet50 Encoder
↓
2048-dimensional image feature vector
↓
Linear projection layer
↓
LSTM Decoder
↓
Fully Connected Output Layer
↓
Generated Caption
```

Detailed architecture:

| Part | Description |
|---|---|
| Feature extractor | Pretrained ResNet50 without final classification layer |
| Image feature size | 2048-dimensional feature vector |
| Feature projection | Linear layer maps image features to embedding dimension |
| Embedding layer | Converts word IDs into dense vectors |
| LSTM decoder | Generates the caption word by word |
| Output layer | Predicts scores for every word in the vocabulary |
| Loss function | CrossEntropyLoss with ignored padding index |
| Optimizer | Adam |

## Training Evaluation

The model was evaluated using:

- Train loss
- Validation loss
- BLEU-1 score
- BLEU-2 score
- Qualitative generated examples

Train loss shows how well the model learns from training data.

Validation loss shows how well the model works on unseen validation data.

BLEU score compares generated captions with human-written reference captions.

## Week 3 Result

The improved model became stronger than the baseline because it used ResNet50, dropout, and better hyperparameters.

The model was able to generate simple captions. However, some captions were still generic because Flickr8k is a small dataset and the decoder used greedy word generation.

---

# Week 4: Model Saving and Streamlit Application

## Goal of Week 4

The goal of Week 4 was to finalize the project, save the trained model, and create a small application for testing the image captioning system.

## Work Completed

During Week 4, the following tasks were completed:

- Saved the improved ResNet50 + LSTM model
- Saved the vocabulary file
- Created a Streamlit application
- Added image upload function
- Added image preprocessing
- Loaded the trained model inside the app
- Generated captions for uploaded images
- Tested the app with a real uploaded image

## Saved Artifacts

The final model files were saved in the `artifacts/` folder.

| Artifact | Description |
|---|---|
| `artifacts/resnet50_lstm_captioner.pt` | Saved trained model checkpoint |
| `artifacts/vocab.json` | Saved vocabulary with word-to-index and index-to-word mappings |

## Streamlit Application

The application allows users to upload an image and generate a caption.

Application components:

| App component | Output / function |
|---|---|
| Model artifact | `artifacts/resnet50_lstm_captioner.pt` |
| Vocabulary artifact | `artifacts/vocab.json` |
| Image preprocessing | Resize to 224 × 224, convert to tensor, normalize |
| Feature extractor | Pretrained ResNet50 in evaluation mode |
| Caption generation | Greedy decoding from `<start>` until `<end>` or max length |
| Maximum caption length | 25 tokens |

## Streamlit App Output

The mini application was tested with an uploaded image.

Visible output from the app screenshot:

```text
A man in a black shirt is a
```

This output proves that the inference pipeline works:

```text
uploaded image → preprocessing → ResNet50 feature extraction → LSTM decoder → generated caption
```

However, the caption is incomplete. This shows one limitation of the model. The model learned a useful beginning of the sentence, but it did not generate a complete and natural caption.

---

# Final Results Summary

| Week | Main Work | Result |
|---|---|---|
| Week 1 | Dataset loading and EDA | Dataset structure and caption length were analyzed |
| Week 2 | Preprocessing and baseline model | VGG16 + LSTM baseline model was created |
| Week 3 | Improved deep learning model | ResNet50 + LSTM with dropout was trained |
| Week 4 | Saving model and app development | Streamlit mini app was created and tested |

---

# Key Findings

The Flickr8k dataset is suitable for a student image captioning project because it contains real images and multiple captions per image.

Splitting by unique images is important because it prevents the same image from appearing in both training and testing.

The baseline VGG16 + LSTM model helped create the first working captioning pipeline.

The improved ResNet50 + LSTM model used a stronger encoder and dropout.

The Streamlit app makes the project practical because the user can upload an image and test the model.

The generated caption from the app was simple and incomplete, so the project still has limitations.

Accuracy is not the main metric for image captioning. BLEU score and qualitative examples are more useful for evaluating generated captions.

---

# Project Structure

```text
DL_project/
│
├── notebooks/
│   └── 00_full_flickr8k_project_week1_to_week4.ipynb
│
├── artifacts/
│   ├── resnet50_lstm_captioner.pt
│   └── vocab.json
│
├── app.py
├── requirements.txt
└── README.md
```

---

# How to Run the Project

## 1. Install Required Libraries

```bash
pip install torch torchvision pandas numpy matplotlib nltk scikit-learn streamlit pillow
```

## 2. Run the Notebook

Open the notebook in Google Colab:

```text
00_full_flickr8k_project_week1_to_week4.ipynb
```

Run all cells from Week 1 to Week 4.

The notebook will:

- Download/load the Flickr8k dataset
- Preprocess captions
- Extract image features
- Train baseline and improved models
- Save model artifacts

## 3. Check Saved Files

After running the notebook, make sure these files exist:

```text
artifacts/resnet50_lstm_captioner.pt
artifacts/vocab.json
```

## 4. Run the Streamlit App

```bash
streamlit run app.py
```

## 5. Upload an Image

After the app opens:

- Upload a JPG, JPEG, or PNG image
- Click `Generate Caption`
- View the generated caption

---

# Example App Result

Input: uploaded image from local computer

Output:

```text
A man in a black shirt is a
```

This result shows that the app works, but also shows that the model needs improvement to generate more complete captions.

---

# Limitations

The model still has several limitations:

- Flickr8k is a small dataset compared with larger datasets such as MS COCO.
- Generated captions can be generic.
- The app output can be incomplete.
- Greedy decoding may choose weak next words.
- The model does not use an attention mechanism.
- Small objects and background details may be missed.
- More training time and stronger models are needed for better caption quality.

---

# Future Work

In the future, this project can be improved by:

- Training for more epochs
- Using beam search instead of greedy decoding
- Adding an attention mechanism
- Using a larger dataset such as Flickr30k or MS COCO
- Trying Transformer-based image captioning models
- Improving the Streamlit app design
- Adding example images inside the app
- Showing several generated caption options

---

# Conclusion

This project successfully developed a complete image captioning system using deep learning.

During Week 1–Week 4, the project went through the full pipeline:

- Dataset loading
- Exploratory data analysis
- Caption preprocessing
- Vocabulary building
- Image feature extraction
- Baseline model training
- Improved model training
- Evaluation
- Artifact saving
- Streamlit application development

The final system uses a pretrained ResNet50 encoder and an LSTM decoder. The application can take an uploaded image and generate a caption.

The final app result showed:

```text
A man in a black shirt is a
```

This result is not perfect, but it proves that the full image captioning pipeline works. The project clearly demonstrates computer vision, natural language processing, transfer learning, sequence modeling, and practical deployment in one complete deep learning project.
