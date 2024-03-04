## Overview
- This project is a Convolutional Autoencoder that colourizes black and white landscape images and brings them to life. The project uses a deep CAE built and trained using PyTorch, streamlit for UI and modelbit for model hosting. You can demo the project [here](https://imagecolourizer-9ymmhc5htmljnzcw5evumt.streamlit.app/) or watch the demo video below.

https://github.com/akashvshroff/ImageColourizer/assets/63399889/5dd8ed4c-5787-47bb-bed5-f5dd1c7b926d

- I undertook this project primarily to get my hands dirty with model hosting and serving. The CAE architecture was thrown together fairly quickly and trained on a large (but slightly lacking), publically-available dataset from Kaggle (see below for more details). 
- Several improvements can be made. Primarily, allowing the model to train on higher quality data would significantly improve its performance and ability to generalize its learnings. More varied training data, i.e. more types of images beyond landscapes, would improve model versatility and overall performance. 

### Data & ML Architecture
- The data for this project was from this Kaggle [dataset](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization) and is a collection of roughly 7000 images in grayscale and colour. The data is a great starting point, however, is lacking in some regards. Images are fairly low-resolution (150x150 pixels) and are not receptive to resolution enhancement (via FSRCNN, see [`data_processing.ipynb`](https://github.com/akashvshroff/ImageColourizer/blob/main/data_processing.ipynb) for my failed attempts at improving quality).
- Moreover, I found that a majority of the images were 'blue biased', meaning were composed of large blue elements like the sky, water bodies etc. This meant that the model generally tended to erroneously predict parts of the image as blue.

- In terms of model architecture, I chose to implement a U-net style architecture with skip connections in order to propagate finer details that would have been otherwise lost during pooling. A great explanation for the role of skip connections in deep learning can be found [here](https://theaisummer.com/skip-connections/).

