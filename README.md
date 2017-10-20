# Simple adversarial attack on InceptionV3

Attack InceptionV3 net using FGM( fast gradient method)  and show saliency maps.

## Prerequisites

- Python 3.5+
- tensorflow 1.2+
- sklearn, numpy, pandas, skimage

## Dataset and model

Download the dataset and model file from Kaggle: https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack/data

+ Download Inception-v3 model ckpt file  and put it in ./ .

+ Download the 1000-images development-set zip file and unzip the *images* directory to ./images, such that the image files are located in ./images/images/.

+ Finally, If you would like to generate and save the saliency maps, uncomment the lines following 

  ```python
  # generate saliency maps
  ```

  in main.py.

  ​

## Acknowledgements

Inspired by Kaggle competitions on adversarial attacks and tensorflow/cleverhans.
