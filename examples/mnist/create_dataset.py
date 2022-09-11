"""Special module for downloading MNIST dataset."""

import argparse
import gzip
import logging
import zipfile
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Special script for MNIST dataset for further training.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="The directory for storing the MNIST dataset."
    )
    parser.add_argument(
        "--valid_size",
        required=True,
        help="The percentage of the train data needs to be treated as validation data."
    )
    return parser.parse_args()


def download(download_dir: Path) -> None:
    """Downloads the MNIST raw files to given directory."""

    download_url('https://data.deepai.org/mnist.zip', download_dir.as_posix())

    downloaded_file = download_dir.joinpath('mnist.zip')
    with zipfile.ZipFile(downloaded_file) as f:
        f.extractall(download_dir)
    downloaded_file.unlink()

    logger.info(f"Raw Files are downloaded in {download_dir}.")


def load_mnist_images(file_path: Union[str, Path]) -> np.ndarray:
    """Loads the images from given MNIST gzip path.

    https://stackoverflow.com/a/62781370/8199034
    """

    with gzip.open(file_path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')

        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')

        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')

        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')

        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))

    logger.info(f"Images are read from {file_path}, shape={images.shape}.")

    return images


def load_mnist_labels(file_path: Union[str, Path]) -> np.ndarray:
    """Loads the labels from given MNIST gzip path.

    https://stackoverflow.com/a/62781370/8199034
    """

    with gzip.open(file_path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')

        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')

        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)

    logger.info(f"Labels are read from {file_path}, shape={labels.shape}.")

    return labels


def write_dataset(output_dir: Path, images: np.ndarray, labels: np.ndarray, mode: str) -> pd.DataFrame:
    """Writes the dataset"""

    assert len(images) == len(labels)

    folder = output_dir.joinpath(mode)
    folder.mkdir(exist_ok=True, parents=True)

    tmp = {}
    collection = []
    for label, image in zip(labels, images):
        label = int(label)
        label_folder = folder.joinpath(f'{label}')
        label_folder.mkdir(exist_ok=True)
        count = tmp.get(label, 0)
        count += 1
        saving_path = str(label_folder.joinpath(f'{count}.png'))
        cv2.imwrite(saving_path, image)
        tmp[label] = count
        collection.append((saving_path, label))

    collection = pd.DataFrame(collection, columns=['image_path', 'label'])
    collection['mode'] = mode
    collection.to_csv(folder.joinpath(f'{mode}_samples.CSV'))
    return collection


def create_dataset(output_dir: str, valid_size: float):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_images_file = output_dir.joinpath('train-images-idx3-ubyte.gz')
    train_labels_file = output_dir.joinpath('train-labels-idx1-ubyte.gz')
    test_images_file = output_dir.joinpath('t10k-images-idx3-ubyte.gz')
    test_labels_file = output_dir.joinpath('t10k-labels-idx1-ubyte.gz')

    if not all(x.exists() for x in (train_images_file, train_labels_file, test_images_file, test_labels_file)):
        download(output_dir)

    test_images = load_mnist_images(test_images_file)
    test_labels = load_mnist_labels(test_labels_file)
    write_dataset(output_dir, test_images, test_labels, 'test')

    train_images = load_mnist_images(train_images_file)
    train_labels = load_mnist_labels(train_labels_file)

    split_index = int(len(train_images) * valid_size)
    valid_images = train_images[:split_index]
    valid_labels = train_labels[:split_index]
    write_dataset(output_dir, valid_images, valid_labels, 'valid')

    train_images = train_images[split_index:]
    train_labels = train_labels[split_index:]
    write_dataset(output_dir, train_images, train_labels, 'train')


if __name__ == '__main__':
    args = parse_args()
    create_dataset(**vars(args))
