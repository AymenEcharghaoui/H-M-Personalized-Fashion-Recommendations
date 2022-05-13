import os
import torch
from skimage import io, transform
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import time
import cloudpickle as pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




class ArticlesDataset(Dataset):

    def __init__(self, images_dir, transactions_dir, transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            transactions_dir (string): Directory of the transactions csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.transform = transform
        df =  pd.read_csv(transactions_dir)
        transactions = {}

        # build relevant articles for each image
        # put all customers
        for i,row in df.iterrows():
            transactions[row['customer_id']] = set()

        # put all transactions
        for i,row in df.iterrows():
            customer = row['customer_id']
            article = row['article_id']
            transactions[customer].add(article)

        # get all relevant articles for each p_article
        self.relevant = {}

        # put all articles
        for i,row in df.iterrows():
            self.relevant[row['article_id']] = set()

        # populate --relevant
        for customer in transactions:
            for p_article in transactions[customer]:
                for article in transactions[customer]:
                    if(article != p_article):
                        self.relevant[p_article].add(article)

        # build an index of all image files
        self.index = {}
        i = 0
        for image in os.listdir(self.images_dir):
            self.index[int(image[:-4])] = i
            i += 1

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):

        article_id = os.listdir(self.images_dir)[idx]
        img_name = os.path.join(self.images_dir,article_id)
        image = io.imread(img_name)

        label_images = self.relevant[int(article_id[:-4])]
        label = torch.zeros(len(self.index),dtype=torch.float32)

        for article in label_images:
            label[self.index[article]] = 1/len(label_images)
        
        if self.transform:
            return (self.transform(image),label)
        
        return (image,label)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size.
    """

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        img = transform.resize(image, self.output_size)
        return img


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

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        img = image[top: top + new_h,
                      left: left + new_w]

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = image.transpose((2, 0, 1))
        return torch.from_numpy(img).type(torch.float32)

class Model(torch.nn.Module):

    def __init__(self,num_articles,activation = torch.nn.ReLU()) :
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv3 = torch.nn.Conv2d(6,16,kernel_size=3,stride=1,padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dense5 = torch.nn.Linear(16*56*56,int(num_articles/2))
        self.dense6 = torch.nn.Linear(int(num_articles/2),num_articles)
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


def trainer(training_generator,model,loss_fn,epoch,rate,train_period) :
    optimizer = torch.optim.Adam(params=model.parameters(),lr=rate,weight_decay=1e-4,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loss = []

    for i in range(epoch):
        running_loss = 0.0
        total = 0
        for j, sample_batched in enumerate(training_generator):
            x,y = sample_batched
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            total += y.size(0)
            running_loss += loss.item() * y.size(0)
            
            if j % train_period == train_period-1:
                print('epoch:%d, period:%d running loss: %.3f' %(i + 1, j + 1, loss.item()))
        
        running_loss = running_loss/total
        print('epoch:%d average loss: %.3f' %(i + 1, running_loss))
        train_loss.append(running_loss)
        
    return train_loss
def score(model,images_dir,num_articles,tr_dir,num_recomm=12,transform=None):
    """
    return MAP@12
    """

def predictions(model,tr_dir,cust_dir,pred_dir,images_dir,num_articles,num_reccom=12,transform=None) :
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
    is_active = {}

    # all customers : there are new customers in customers.csv
    customers = pd.read_csv(cust_dir)
    for i,row in customers.iterrows():
        recommandations[row['customer_id']] = torch.zeros(num_articles,dtype=torch.float32)
        is_active[row['customer_id']] = False

    # making recommandations based on previous transactions
    transactions = pd.read_csv(tr_dir)
    for i,row in transactions.iterrows():

        assert row['customer_id'] in recommandations

        is_active[row['customer_id']] = True
        image_id = row['article_id']
        img_name = os.path.join(images_dir,'0'+str(image_id)+'.jpg')
        image = io.imread(img_name)
        if transform:
            image = transform(image)
        image = image.to(device)
        recommandations[row['customer_id']] += model(image.unsqueeze(0)).squeeze(0).to('cpu')

    submission_file = open(pred_dir,'w')
    # no worries of a second execution : we overwrite what's already existing in the submission file

    submission = csv.writer(submission_file,delimiter=',')

    submission.writerow(['customer_id','prediction'])

    for i,row in customers.iterrows():
        line = [row['customer_id']]
        articles = ""
        reccs = recommandations[row['customer_id']]

        if(not(is_active[row['customer_id']])):
            # new customer : generate num_reccom random articles
            for _ in range(num_reccom-1):
                articles += os.listdir(images_dir)[random.randint(0,num_articles)][:-4]+" "
            articles += os.listdir(images_dir)[random.randint(0,num_articles)][:-4]
        else:
            indices = reccs.topk(num_reccom).indices
            for i in range(num_reccom-1):
                articles += os.listdir(images_dir)[indices[i]][:-4]+ " "
            articles += os.listdir(images_dir)[indices[num_reccom-1]][:-4]

        line.append(articles)
        submission.writerow(line)

    submission_file.close()
    
def lossPlot(loss):
    plt.plot(loss,label = "loss for training set")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./loss_curves.png")
    
    file = open("./loss.txt", "w")
    for element in loss:
        file.write(element + "\n")
    file.close()

if __name__ == '__main__':
    start_time = time.time()
    print(torch.cuda.is_available())
    images_dir = '~/data/images__all/'
    transactions_dir = '~/data/transactions_train.csv'
    transactions_dir_train = '~/data/transactions_train_train.csv'
    transactions_dir_test = '~/data/transactions_train_test.csv'
    customers_dir = '~/data/customers.csv'
    predictions_dir='~/data/submission.csv'
    '''
    images_dir = './data/images/images_test/'
    transactions_dir = './data/transactions_train_10.csv'
    customers_dir = './data/customers_10.csv'
    predictions_dir='./data/submission_10.csv'
    '''
    batch_size = 64
    train_period = 10
    num_recomm = 12
    num_articles = len(os.listdir(images_dir)) #105100
    myTransform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])
    dataset = ArticlesDataset(images_dir = images_dir,transactions_dir = transactions_dir_train,transform=myTransform)
    print("dataset : --- %s seconds ---" % (time.time() - start_time))
    training_generator = DataLoader(dataset, batch_size = batch_size,shuffle = True, num_workers = 0)
    model = Model(num_articles=num_articles)
    if(torch.cuda.is_available()):
        model.cuda()
    train_loss = trainer(training_generator,model,torch.nn.CrossEntropyLoss(),epoch = 10,rate = 1e-2, train_period=train_period)
    torch.save(model.state_dict(), "model.pt")
    lossPlot(train_loss)
    print("training : --- %s seconds ---" % (time.time() - start_time))

    '''
    model0 = Model(num_articles=num_articles)
    model0.load_state_dict(torch.load("model.pt"))
    model0.eval()
    '''
    predictions(model,num_reccom=num_recomm,tr_dir=transactions_dir_train,cust_dir=customers_dir,pred_dir=predictions_dir,images_dir=images_dir,num_articles=num_articles,transform=myTransform)
    print("making predictions : --- %s seconds ---" % (time.time() - start_time))
