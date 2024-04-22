# CV Project Overview

Author: Danylo Vanin

This project implements a deep learning model that uses a combination of CNN and LSTM to generate textual descriptions of images. The model is trained on the Flickr8k dataset, which consists of 8,000 images each paired with five different captions. The implementation uses the VGG16 architecture to extract features from the images, and an LSTM network to generate captions based on these features.

## Requirements

- Python 3.x
- Keras
- NumPy
- Matplotlib
- IPython
- Jupyter Notebook or similar Python environment

This code is tested with Python 3.8, but should be compatible with other versions that support the libraries listed.

## Setup and Installation

First, ensure that Python 3.x and pip are installed on your system. You can then install the required Python libraries using pip:

```bash
pip install keras numpy matplotlib ipython jupyter
```

## Dataset Download

The dataset used is the Flickr8k dataset, which can be downloaded using the following commands. These commands fetch the dataset and accompanying annotations, unzip them, and clean up the downloaded zip files.

```bash
wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
unzip -qq Flickr8k_Dataset.zip
unzip -qq Flickr8k_text.zip
rm Flickr8k_Dataset.zip Flickr8k_text.zip
```

Ensure you have sufficient storage and network conditions for downloading and unzipping the dataset which includes thousands of images and captions.

## Running the Code

1. **Start your Jupyter Notebook or Python environment** where you can run `.ipynb` or `.py` files.
2. **Load the script** provided in the repository.
3. **Execute the script** which is self-contained. It will process the images, train the model, and provide output directly in your Python environment.

The script will display an image from the dataset, process all image captions, extract features from images using a pre-trained VGG16 model, prepare sequences, and finally, train a neural network to generate captions. Model progress will be plotted after training, showing loss and accuracy over epochs.

## Example Usage

After running the script, you can use the model to generate captions for new images by calling the `generate_desc` function:

```python
photo_features = extract_features('path_to_your_image.jpg')
description = generate_desc(model, tokenizer, photo_features, max_length)
print("Generated Description:", description)
```

Replace `'path_to_your_image.jpg'` with the actual path to your image.