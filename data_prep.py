import os
import torch
from skimage import io, transform
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import time
import cloudpickle as pickle

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def creatDataset(images_dir, articles_dir,transactions_dir, transform=None):
    """
    Group the articles by [4 groups of Garment Upper body: [Sweater:9302,
    T-shirt and Vest top:7904+2991, Top and Blouse and Jacket:4155+3979+3940, 
    rest:3405+2356+1550+1110+913+460+449+154+73], 2 groups of Garment Lower body:
    [Trousers:11169, rest:3939+2696+1878+130], Garment Full body:13292, 
    Accessories:11158, Shoes and Socks & Tights and Nightwear:5283+2442+1899, 
    rest:5490+3127+121+54+49+25+17+13+9+5+3+2]

    Parameters
    ----------
    images_dir : str
        Directory with all the images.
    articles_dir : str
        Directory of the articles csv file.
    transactions_dir : str
        Directory of the transactions csv file.
    transform : callable, optional
        Optional transform to be applied on a sample.

    Returns
    -------
    id2group: dic {article_id: (group_id, id_in_group)}
    group2id: dic {(group_id, id_in_group): article_id}
    group_sizes: list of group size(#Ai)
    datasets: list of dataset
    """
    
    # creat transactions dictionary {customer: a set of all article he has bought}
    transactions_df = pd.read_csv(transactions_dir, usecols=['customer_id','article_id'], dtype={'article_id':str})
    
    transactions = {}
    for i,row in transactions_df.iterrows():
        customer = row['customer_id']
        article = row['article_id']
        if customer not in transactions:
            transactions[customer] = {article}
        else:
            transactions[customer].add(article)
            
    # creat group2id dir and id2group and group_sizes
    (group2id,id2group,group_sizes) = creatArticlesDic(articles_dir)
    
    # creat a set of all articles_id with image
    images_set = set([image[:-4] for image in os.listdir(images_dir)])
    
    # creat relevant dir {article: a set of all articles(1d2group[article]) relevant to it}
    relevant = {}
    id_relevant = []
    for customer in transactions:
        for p_article in transactions[customer]:
            if p_article in images_set:
                for article in transactions[customer]:
                    if p_article != article:
                        if p_article not in relevant:
                            relevant[p_article] = {id2group[article]}
                            id_relevant.append(p_article)
                        else:
                            relevant[p_article].add(id2group[article])
    # creat datasets
    datasets = [] #list of datasets
    for i in range(len(group_sizes)):
        datasets.append(ArticlesDataset(i,images_dir,group_sizes[i],relevant,id_relevant,transform=transform))
        
    return (group2id,id2group,group_sizes,datasets)

def creatArticlesDic(articles_dir):
    articles_df = pd.read_csv(articles_dir, usecols=['article_id','product_type_name','product_group_name'], dtype={'article_id':str})
    
    group_sizes = [0]*10 #list of group sizes(#Ai)
    
    group2id = {}
    id2group = {}
    for i,row in articles_df.iterrows():
        article = row['article_id']
        
        if row['product_group_name'] == 'Garment Upper body':
            if row['product_type_name'] == 'Sweater':
                group_id = 0
            elif row['product_type_name'] == 'T-shirt' or \
                row['product_type_name'] == 'Vest top':
                group_id = 1
            elif row['product_type_name'] == 'Top' or \
                row['product_type_name'] == 'Blouse' or \
                row['product_type_name'] == 'Jacket':
                group_id = 2
            else:
                group_id = 3
        
        elif row['product_group_name'] == 'Garment Lower body':
            if row['product_type_name'] == 'Trousers':
                group_id = 4
            else:
                group_id = 5
                
        
        elif row['product_group_name'] == 'Garment Full body':
            group_id = 6
        
        elif row['product_group_name'] == 'Accessories':
            group_id = 7
        
        elif row['product_group_name'] == 'Shoes' or \
            row['product_group_name'] == 'Socks & Tights' or \
            row['product_group_name'] == 'Nightwear':
            group_id = 8
            
        else:
            group_id = 9
        
        id2group[article] = (group_id,group_sizes[group_id])
        group2id[(group_id,group_sizes[group_id])] = article
        group_sizes[group_id] += 1
        
    return (group2id,id2group,group_sizes)

class ArticlesDataset(Dataset):

    def __init__(self, group_id, images_dir, group_size, relevant, id_relevant, transform=None):
        self.group_id = group_id
        self.images_dir = images_dir
        self.group_size = group_size
        self.relevant = relevant
        self.id_relevant = id_relevant
        self.transform = transform
        
        self.length = len(id_relevant)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        article_id = self.id_relevant[idx]
        img_name = os.path.join(self.images_dir,article_id+'.jpg')
        image = io.imread(img_name)

        label_images = self.relevant[article_id]
        label = torch.zeros(self.group_size, dtype=torch.float32)

        for (group_id, id_in_group) in label_images:
            if(group_id == self.group_id):
                label[id_in_group] = 1
        
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

    def __init__(self,group_length,activation = torch.nn.ReLU()) :
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv3 = torch.nn.Conv2d(6,16,kernel_size=3,stride=1,padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.dense5 = torch.nn.Linear(16*56*56,int(group_length/2))
        self.dense6 = torch.nn.Linear(int(group_length/2),group_length)
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
def score(tr_dir,pred_dir,num_recomm=12):
    """
    return MAP@12 over the test dataset
    Args : 
        tr_dir (string): directory of transactions_train_test.csv
        pred_dir (string): Directory of the submission file
    """
    # build from transactions_test
    df =  pd.read_csv(tr_dir,dtype={'customer_id':str,'article_id':str})
    transactions = {}
    # put all customers
    for i,row in df.iterrows():
        transactions[row['customer_id']] = set()
    # put all transactions
    for i,row in df.iterrows():
        customer = row['customer_id']
        article = row['article_id']
        transactions[customer].add(article)
    # computing MAP@12

    map12 = 0
    subm = open(pred_dir)
    subm_reader = csv.reader(subm)
    next(subm_reader)
    count_customers = 0
    for row in subm_reader:
        count_customers += 1
        # going through all customers
        average_precision = 0
        reccomandations = row[1].split()
        customer_id = row[0]
        relevant = 0
        for i in range(num_recomm):
            if(reccomandations[i] in transactions[customer_id]):
                relevant += 1
                average_precision += relevant/(i+1)
        average_precision /= min(12,len(transactions[customer_id]))
        map12 += average_precision
    map12 /= count_customers
    return map12


def predictions(models,id2group,group2id,group_sizes,tr_dir,cust_dir,pred_dir,images_dir,num_articles,num_reccom=12,transform=None) :
    """
    store a sample submission csv file in pred_dir
    Args :
        models : DL models for each group of articles after being trained on the whole dataset
        id2group (dict): article_id -> (group_index,index in group) 
        group2id (dict): (group_index,index in group) -> article_id
        group_sizes (list): group_index -> size of this group
        tr_dir (string): directory of transactions_train.csv
        cust_dir (string): directory of customers.csv
        pred_dir (string): directory of the submission file
        images_dir (string): Directory with all the images.
        num_articles (int): total number of articles
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group_sizes_cumm = []
    start = 0
    for size in group_sizes:
        start += size
        group_sizes_cumm.append(start)

    assert group_sizes_cumm[-1] == num_articles

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
        group_index = id2group[image_id][0]
        img_name = os.path.join(images_dir,'0'+str(image_id)+'.jpg')
        image = io.imread(img_name)
        if transform:
            image = transform(image)
        image = image.to(device)
        #start = sum(group_sizes[:id2group[image_id][0]])
        #end = start + group_sizes[id2group[image_id][0]]
        end = group_sizes_cumm[id2group[image_id][0]]
        start = end - group_sizes[id2group[image_id][0]]
        recommandations[row['customer_id']][start:end] += models[group_index](image.unsqueeze(0)).squeeze(0).to('cpu')

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
            '''
            for i in range(num_reccom-1):
                articles += os.listdir(images_dir)[indices[i]][:-4]+ " "
            articles += os.listdir(images_dir)[indices[num_reccom-1]][:-4]
            '''
            for i in range(num_reccom):
                index = indices[i]
                j = 0
                while(index<=group_sizes_cumm[j]):
                    j += 1
                # image in group j
                if(j>0):
                    index -= group_sizes_cumm[j-1]
                # image index in group j is index
                articles += group2id[(j,index)]+ " "
            articles  = articles[:-1]

        line.append(articles)
        submission.writerow(line)

    submission_file.close()

    
def lossPlot(loss,dir,i):
    plt.plot(loss,label = "loss for training set")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dir+"loss_curves_"+str(i)+".png")
    
    file = open(dir+"loss"+str(i)+".txt", "w")
    for element in loss:
        file.write(str(element) + "\n")
    file.close()

def init_weights_xavier_uniform(m):
    if type(m) == torch.nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    start_time = time.time()
    print('Is cuda available?', torch.cuda.is_available())
    
    images_dir = '/home/aymen/data/images__all/'
    transactions_dir = '/home/aymen/data/transactions_train.csv'
    transactions_dir_train = '/home/aymen/data/transactions_train_train.csv'
    transactions_dir_test = '/home/aymen/data/transactions_train_test.csv'
    articles_dir = '/home/aymen/data/articles.csv'
    customers_dir = '/home/aymen/data/customers.csv'
    predictions_dir = '/home/aymen/data/submission.csv'
    loss_dir = '/home/aymen/data/'
    '''
    images_dir = './data/images/images_test/'
    transactions_dir_train = './data/transactions_train_10.csv'
    articles_dir = './data/articles.csv'
    customers_dir = './data/customers_10.csv'
    predictions_dir='./data/submission_10.csv'
    loss_dir = './data/'
    '''
    
    batch_size = 64
    epoch = 10
    rate = 1e-3
    train_period = 10
    num_recomm = 12
    
    num_articles = len(os.listdir(images_dir)) #105100
    myTransform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])
    
    (group2id,id2group,group_sizes,datasets) = creatDataset(images_dir, articles_dir, transactions_dir_train, transform = myTransform)
    print("creating dataset : --- %s seconds ---" % (time.time() - start_time))
    
    models = []
    for i in range(len(group_sizes)):

        model_submit_dir = './data/model'+str(i)+'.pt'

        dataset = datasets[i]
        print("dataset "+str(i)+": --- %s seconds ---" % (time.time() - start_time))
        training_generator = DataLoader(dataset, batch_size = batch_size,shuffle = True, num_workers = 2)
        model = Model(group_length=group_sizes[i])
        model.apply(init_weights_xavier_uniform)
        if(torch.cuda.is_available()):
            model.cuda()
        train_loss = trainer(training_generator,model,torch.nn.BCEWithLogitsLoss(),epoch=epoch,rate=rate, train_period=train_period)
        torch.save(model.state_dict(), model_submit_dir)
        lossPlot(train_loss,loss_dir,i)
        print("training "+str(i)+": --- %s seconds ---" % (time.time() - start_time))
        model.to(torch.device('cpu'))
        torch.cuda.empty_cache()
        models.append(model)

    '''
    model0 = Model(num_articles=num_articles)
    model0.load_state_dict(torch.load("model.pt"))
    model0.eval()
    '''
    predictions(models,id2group=id2group,group2id=group2id,group_sizes=group_sizes,num_reccom=num_recomm,tr_dir=transactions_dir_train,cust_dir=customers_dir,pred_dir=predictions_dir,images_dir=images_dir,num_articles=num_articles,transform=myTransform)
    print("making predictions : --- %s seconds ---" % (time.time() - start_time))
    map12 = score(tr_dir=transactions_dir_train,pred_dir=predictions_dir,num_recomm=num_recomm)
    print("map12 score is",map12)

# ======Biao's test=========
# (group2id,id2group,group_sizes,datasets) = creatDataset('./data/images/images_test/', './data/articles.csv', './data/transactions_train_10.csv', transform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))
# print(datasets[0][0][0][1][11][11])

# (group2id,id2group,group_sizes) = creatArticlesDic('./data/articles.csv')
# print(group_sizes)
# print(id2group)
