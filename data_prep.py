import os
import torch
from skimage import io, transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




class ArticlesDataset(Dataset):

    def __init__(self, images_dir,transactions_dir, transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            transactions_dir (string): Directory of the transactions csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = root_dir
        self.transform = transform
        df =  pd.read_csv(transactions_dir)
        transactions = {}

        # build relevant articles for each image
        # put all customers
        for i,row in df.iterrows():
            transactions[row['customer_id']] = []

        # put all transactions
        for i,row in df.iterrows():
            customer = row['customer_id']
            article = row['article_id']
            transactions[customer].append(article)

        # get all relevant articles for each p_article
        self.relevant = {}

        # put all articles
        for i,row in df.iterrows():
            self.relevant[row['article_id']] = {}

        # populate --relevant
        for customer in transactions:
            for p_article in transactions[customer]:
                for article in transactions[customer]:
                    if(article != p_article):
                        self.relevant[p_article][article]=0

        # build an index of all image files
        self.index = {}
        i = 0
        for image in os.listdir(self.root_dir):
            self.index[image[:-4]] = i
            i += 1

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):

        article_id = os.listdir(self.root_dir)[idx]
        img_name = os.path.join(self.root_dir,
                                article_id+'.jpg')
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        label_images = self.relevant[article_id]
        label = torch.zeros(len(self.index),device = device,dtype=torch.float64)
        for i in self.index[article_id]:
            label[i] = 1/i

        return {'image':image,'label':label}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size.
    """

    def __init__(self, output_size):

        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):

        image,label = sample['image'],sample['label']
        img = transform.resize(image, self.output_size)
        return {'image':img,'label':label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        image,label = sample['image'],sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h)
        left = torch.randint(0, w - new_w)

        img = image[top: top + new_h,
                      left: left + new_w]

        return {'image':img,'label':label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image,label = sample['image'],sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = image.transpose((2, 0, 1))
        return {'image':img,'label':label}


# dataloader
dataset = ArticlesDataset(root_dir='~/data/images__all/',transactions_dir = '~/data/transactions_train.csv',transform=transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))
dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0)