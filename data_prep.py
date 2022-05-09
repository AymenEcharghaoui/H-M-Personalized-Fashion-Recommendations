import os
import torch
from skimage import io, transform
import pandas as pd
import numpy as np
import csv
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
        self.images_dir = images_dir
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
        for image in os.listdir(self.images_dir):
            self.index[image[:-4]] = i
            i += 1

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):

        article_id = os.listdir(self.images_dir)[idx]
        img_name = os.path.join(self.images_dir,article_id)
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        label_images = self.relevant[article_id[:-4]]
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image,label = sample['image'],sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = image.transpose((2, 0, 1))
        return {'image':torch.from_numpy(img).to(device),'label':torch.from_numpy(label).to(device)}

class Model(nn.Module):

    def __init__(self,num_articles,activation = torch.nn.ReLU()) :
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,6,kernel_size=1,stride=1,padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv3 = torch.nn.Conv2d(6,16,kernel_size=1,stride=1,padding=0)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dense5 = torch.nn.Linear(16*56*56,num_articles/2)
        self.dense6 = torch.nn.Linear(num_articles/2,num_articles)
        self.activation = activation

    def forward(self,x) :
        z = self.conv1(x)
        z = self.pool2(z)
        z = self.conv3(z)
        z = self.pool4(z)
        z = z.view(z.size(0),-1)
        z = self.dense5(z)
        z = self.activation(z)
        z = self.dense6(z)
        return z

def trainer(training_generator,model,loss_fn,epoch,batch_size,rate) :
    optimizer = torch.optim.Adam(params=model.parameters(),lr=rate,weight_decay=1e-4)

    for _ in range(epoch):
        for sample_batched in training_generator:
            x = sample_batched['image']
            y = sample_batched['label']
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

def predictions(model,tr_dir,cust_dir,pred_dir,images_dir,num_articles,transform=None) :
    """
    store a sample submission csv file in pred_dir
    Args :
        model : DL model after being trained on the whole dataset
        tr_dir (string): directory of transactions_train.csv
        cust_dir (string): directory of customers.csv
        pred_dir (string): directory of the submission file
        images_dir (string): Directory with all the images.
        num_articles (int): total number of articles
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recommandations = {}

    # all customers : there are new customers in customers.csv
    customers = pd.read_csv(cust_dir)
    for i,row in customers.iterrows():
        recommandations[customers['id']] = torch.zeros(num_articles,device = device,dtype=torch.float64)

    # making recommandations based on previous transactions
    transactions = pd.read_csv(tr_dir)
    for i,row in transactions.iterrows():

        assert row['customer_id'] in recommandations

        image_id = row['article_id']
        img_name = os.path.join(images_dir,image_id+'.jpg')
        image = io.imread(img_name)
        if transform:
            image = transform(image)

        recommandations[row['customer_id']] += model(image)

    submission_file = open(pred_dir,'w')
    # no worries of a second execution : we overwrite what's already existing in the submission file

    submission = csv.writer(submisson_file,delimiter=',')

    for i,row in customers.iterrows():
        line = [row['customer_id']]
        articles = ""
        reccs = recommandations[row['customer_id']]

        if(torch.equal(reccs,torch.zeros(num_articles,device = device,dtype=torch.float64))):
            # new customer : generate 12 random articles
            for _ in range(11):
                articles += os.listdir(images_dir)[random.randint(0,num_articles)][:-4]+" "
            articles += os.listdir(images_dir)[random.randint(0,num_articles)][:-4]
        else:
            indices = reccs.topk(12).indices
            for i in range(11):
                articles += os.listdir(images_dir)[indices[i]]+ " "
            articles += os.listdir(images_dir)[indices[11]]

        line.append(articles)
        submission.writerow(line)

    submission_file.close()


if __name__ == '__main__':
    batch_size = 16
    num_articles = len(os.listdir('~/data/images__all/')) #105100
    transform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])
    dataset = ArticlesDataset(images_dir = '~/data/images__all/',transactions_dir = '~/data/transactions_train.csv',transform=transform)
    training_generator = DataLoader(dataset, batch_size = batch_size,shuffle = True, num_workers = 0)
    model = Model()
    trainer(training_generator,model,torch.nn.CrossEntropyLoss(),epoch = 5,batch_size = 16,rate = 1e-2)
    predictions(model,tr_dir='~/data/transactions_train.csv',cust_dir='~/data/customers.csv',pred_dir='~/data/submission.csv',images_dir='~/data/images__all/',num_articles=num_articles,transform=transform)