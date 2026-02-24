from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import torch

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def load_galaxy_data(label_dir, image_dir):
    """
    Load galaxy images and their corresponding labels from a DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing image filenames and labels.
    image_dir (str): Directory where the images are stored.

    Returns:
    list: A list of tuples containing image paths and their corresponding labels.
    """
    df = pd.read_csv(label_dir)

    data = []
    for index, row in df.iterrows():
        image_path = f"{image_dir}\\{row['GalaxyID']}"[:-2]+".jpg"
        label = row.drop('GalaxyID')
        data.append((image_path, label))
    return data

data = load_galaxy_data(r'C:\Users\olive\Documents\python_codes\a_github_collection\galaxy_morphology_classification\galaxy-zoo-the-galaxy-challenge\training_solutions_rev1\training_solutions_rev1.csv', r'C:\Users\olive\Documents\python_codes\a_github_collection\galaxy_morphology_classification\galaxy-zoo-the-galaxy-challenge\images_training_rev1')

def get_galaxy_batch(data, batch_size=None, transform=None):
    """
    Load a batch of galaxy images by indices.

    Args:
        df: DataFrame with 'path' and 'label' columns
        batch_size: number of samples to load (if None, load all). If batch_size not None, sampling is random.
        transform: Optional image transformation function

    Returns:
        Tuple of (images, labels) as tensors
    """
    to_tensor = transforms.ToTensor()

    if batch_size is None:
        batch_data = data
    else:
        batch_data = random.sample(data, batch_size)

    images = []
    labels = []

    for image_path, label in batch_data:
        image = Image.open(image_path).convert("RGB")

        if transform:
            image = transform(image)

        image = to_tensor(image)
        images.append(image)
        labels.append(torch.tensor(label.values, dtype=torch.float32))

    return torch.stack(images), torch.stack(labels)

# # ml_data = get_galaxy_batch(data, batch_size=32, transform=transforms.Grayscale(num_output_channels=1))
# #
# # plt.figure()
# # plt.imshow(ml_data[0][0][0], cmap='gray')
#
# ml_data = get_galaxy_batch(data, batch_size=32)
#
# plt.figure()
# img = np.transpose(ml_data[0][0])
# plt.imshow(img)   # img is your RGB array
# plt.axis('off')   # hides axis ticks
# plt.show()