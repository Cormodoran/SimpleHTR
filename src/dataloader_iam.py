import pickle
import random
from collections import namedtuple
from typing import Tuple
from matplotlib import pyplot as plt

import numpy as np
from path import Path
import pickle

Sample = namedtuple('Sample', 'imgs, gt_text')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 data_dir: Path,
                 batch_size: int,
                 data_split: float = 0.8) -> None:
        """Loader for dataset."""

        assert data_dir.exists()

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        sentences_to_read = open(data_dir / "sentences.pickle", "rb")
        self.sentences = pickle.load(sentences_to_read)
        sentences_to_read.close()

        images_to_read = open(data_dir / "images.pickle", "rb")
        self.images = pickle.load(images_to_read)
        images_to_read.close()
        
        flat_list = [item for sublist in self.sentences for item in sublist]
        flat_list = ''.join(flat_list)
        chars = set(flat_list)

        for i in range(len(self.sentences)):
            self.samples.append(Sample(self.images[i], self.sentences[i]))


        # split into training and validation set: 95% - 5%
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]


        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            print(self.curr_idx, self.batch_size, len(self.samples))
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller


    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self.samples[i][0] for i in batch_range]
        gt_texts = [self.samples[i][1] for i in batch_range]


        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))
