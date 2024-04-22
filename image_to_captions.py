import os
import string
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, add
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from IPython.display import Image

# Constants
IMAGE_DATASET_PATH = '../data/Flickr8k_Dataset/Flicker8k_Dataset'
CAPTION_DATASET_PATH = '../data/Flickr8k_text/Flickr8k.token.txt'
IMAGE_FILE_PATH = '../data/Flickr8k_Dataset/Flicker8k_Dataset/1000268201_693b08cb0e.jpg'

# Displaying an image
display_image = Image(IMAGE_FILE_PATH)
display(display_image)

def load_caption_file(path):
    """Load and return captions as a dictionary."""
    captions_dict = {}
    with open(path, 'r') as file:
        for line in file:
            tokens = line.split()
            image_id = tokens[0].split('.')[0]
            captions_dict[image_id] = ' '.join(tokens[1:])
    return captions_dict

def clean_captions(captions_dict):
    """Clean captions and return a dictionary with cleaned captions."""
    table = str.maketrans('', '', string.punctuation)
    cleaned_captions = {}
    for image_id, caption in captions_dict.items():
        tokens = caption.split()
        tokens = [word.lower().translate(table) for word in tokens if len(word) > 1]
        cleaned_caption = 'startseq ' + ' '.join(tokens) + ' endseq'
        cleaned_captions[image_id] = cleaned_caption
    return cleaned_captions

def extract_features(directory, image_ids):
    """Extract features using VGG16 for images in the specified directory."""
    model = VGG16(include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for image_id in image_ids:
        filename = f'{directory}/{image_id}.jpg'
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features[image_id] = model.predict(image, verbose=0)
    return features

def prepare_sequences(tokenizer, max_length, captions_dict, features_dict):
    """Prepare sequences of images, input sequences and output words for an entire dataset."""
    X1, X2, y = [], [], []
    vocab_size = len(tokenizer.word_index) + 1
    for key, desc in captions_dict.items():
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(features_dict[key][0])
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def split_data(X1, X2, y, train_size=0.8):
    """Split the data into training and validation sets."""
    total_samples = len(X1)
    train_samples = int(train_size * total_samples)
    return (X1[:train_samples], X2[:train_samples], y[:train_samples]), (X1[train_samples:], X2[train_samples:], y[train_samples:])

def create_tokenizer(captions):
    """Create and fit a tokenizer given caption descriptions."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

def define_model(vocab_size, max_length):
    """Define the deep learning model."""
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

def train_model(model, X1, X2, y, val_data, epochs=20, batch_size=256):
    """Train the model."""
    filepath = './model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    history = model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=val_data, callbacks=[checkpoint])
    return history

def plot_training(history):
    """Plot training and validation loss."""
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    captions_dict = load_caption_file(CAPTION_DATASET_PATH)
    cleaned_captions = clean_captions(captions_dict)
    image_ids = list(cleaned_captions.keys())
    features = extract_features(IMAGE_DATASET_PATH, image_ids[:7000])

    tokenizer = create_tokenizer(cleaned_captions.values())
    max_length = max(len(c.split()) for c in cleaned_captions.values())
    vocab_size = len(tokenizer.word_index) + 1

    X1, X2, y = prepare_sequences(tokenizer, max_length, cleaned_captions, features)
    (train_X1, train_X2, train_y), (val_X1, val_X2, val_y) = split_data(X1, X2, y, train_size=0.8)

    model = define_model(vocab_size, max_length)
    history = train_model(model, train_X1, train_X2, train_y, val_data=(val_X1, val_X2, val_y), epochs=20, batch_size=32)
    plot_training(history)

    # Generate description for the image in the set
    photo_features = features['1234567']
    description = generate_desc(model, tokenizer, photo_features, max_length)
    print("Generated Description:", description)