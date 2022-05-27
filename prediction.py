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
import pickle 

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


def createDataset(images_dir,id2group,group_sizes,transactions_dir, transform=None):
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
    datasets: list of dataset
    """
    
    # create transactions dictionary {customer: a set of all articles he has bought}
    transactions_df = pd.read_csv(transactions_dir, usecols=['customer_id','article_id'], dtype={'article_id':str})
    
    transactions = {}
    for i,row in transactions_df.iterrows():
        customer = row['customer_id']
        article = row['article_id']
        if customer not in transactions:
            transactions[customer] = {article}
        else:
            transactions[customer].add(article)
    
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
        
    return datasets,relevant,id_relevant

def createArticlesDic(articles_dir):
    # based on all articles in articles.csv

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
        self.conv1 = torch.nn.Conv2d(3,4,kernel_size=3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        self.conv3 = torch.nn.Conv2d(4,5,kernel_size=3,stride=1,padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        self.dense5 = torch.nn.Linear(5*14*14,int(group_length/2))
        self.dense6 = torch.nn.Linear(int(group_length/2),group_length)
        self.activation = activation

    def forward(self,x) :
        z = self.conv1(x)
        z = self.activation(z)
        z = self.pool2(z)
        z = self.conv3(z)
        z = self.activation(z)
        z = self.pool4(z)
        z = z.view(z.size(0),-1)
        z = self.dense5(z)
        z = self.activation(z)
        z = self.dense6(z)

        return z

def saveDatasets(group2id,id2group,group_sizes,relevant,id_relevant):
    with open("group2id.pkl", "wb") as f:
        pickle.dump(group2id,f,protocol=pickle.HIGHEST_PROTOCOL)
    with open("id2group.pkl", "wb") as f:
        pickle.dump(id2group,f,protocol=pickle.HIGHEST_PROTOCOL)
    with open("group_sizes.pkl", "wb") as f:
        pickle.dump(group_sizes,f,protocol=pickle.HIGHEST_PROTOCOL)
    with open("relevant.pkl", "wb") as f:
        pickle.dump(relevant,f,protocol=pickle.HIGHEST_PROTOCOL)
    with open("id_relevant.pkl", "wb") as f:
        pickle.dump(id_relevant,f,protocol=pickle.HIGHEST_PROTOCOL)
        
def loadDatasets(images_dir, transform=None):
    with open("group2id.pkl", "rb") as f:
        group2id = pickle.load(f)
    with open("id2group.pkl", "rb") as f:
        id2group = pickle.load(f)
    with open("group_sizes.pkl", "rb") as f:
        group_sizes = pickle.load(f)
    with open("relevant.pkl", "rb") as f:
        relevant = pickle.load(f)
    with open("id_relevant.pkl", "rb") as f:
        id_relevant = pickle.load(f)
    
    datasets = [] #list of datasets
    for i in range(len(group_sizes)):
        datasets.append(ArticlesDataset(i,images_dir,group_sizes[i],relevant,id_relevant,transform=transform))
    
    return (group2id,id2group,group_sizes,datasets)

def score(tr_dir,pred_dir,num_recomm=12):
    """
    return MAP@12 over the test dataset
    Args : 
        tr_dir (string): directory of transactions_train_valid.csv
        pred_dir (string): Directory of the submission file
    """
    # build from transactions_test
    transactions_df =  pd.read_csv(tr_dir,usecols=['customer_id','article_id'],dtype={'article_id':str})
    transactions = {}
    for i,row in transactions_df.iterrows():
        customer = row['customer_id']
        article = row['article_id']
        if customer not in transactions:
            transactions[customer] = {article}
        else:
            transactions[customer].add(article)
    
    # computing MAP@12
    map12 = 0
    subm = open(pred_dir)
    subm_reader = csv.reader(subm)
    next(subm_reader)
    count_customers = 0
    for row in subm_reader:
        customer = row[0]
        if(customer in transactions):
            count_customers += 1
            # going through all customers
            average_precision = 0
            reccomandations = row[1].split()
            relevant = 0
            for i in range(num_recomm):
                if(reccomandations[i] in transactions[customer]):
                    relevant += 1
                    average_precision += relevant/(i+1)
            average_precision /= min(12,len(transactions[customer]))
            if average_precision > 0:
                print(customer+"  GT values : "+str(transactions[customer])+" Predictions : "+str(reccomandations))
            map12 += average_precision
    map12 /= count_customers
    return map12

def predictions(models,id2group,group2id,group_sizes,tr_dir_train,tr_dir_valid,pred_dir,images_dir,num_reccom=12,transform=None) :
    
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
    """
    begin_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(len(models)):
        models[i].eval()
        models[i].to(device)
    
    group_sizes_cumm = []
    start = 0
    for size in group_sizes:
        start += size
        group_sizes_cumm.append(start)

    num_articles = group_sizes_cumm[-1]

    # looking at only a subset of customers
    # create transactions dictionary {customer: a set of all articles he has bought}
    transactions_df = pd.read_csv(tr_dir_train, usecols=['customer_id','article_id'], dtype={'article_id':str})
    
    transactions = {}
    for i,row in transactions_df.iterrows():
        customer = row['customer_id']
        article = row['article_id']
        if customer not in transactions:
            transactions[customer] = {article}
        else:
            transactions[customer].add(article)
    
    print("creat transaction dic, time:",time.time()-begin_time)

    submission_file = open(pred_dir,'w',newline='')
    submission = csv.writer(submission_file,delimiter=',')
    submission.writerow(['customer_id','prediction'])
    
    
    customers_df = pd.read_csv(tr_dir_train, usecols=['customer_id'])
    customers_list = customers_df['customer_id'].unique()
    
    num_new_customer = 0
    for k,customer in enumerate(customers_list):
        
        if customer not in transactions:
            # new customer : do nothing
            num_new_customer += 1
        else:
            # a customer seen in transactions
            with torch.no_grad():
                recommandation = torch.zeros(num_articles,dtype=torch.float32)
                recommandation.to(device)
                print("recom:",recommandation.device)
                
                for article in transactions[customer]:
                    img_name = os.path.join(images_dir, article + '.jpg')
                    if os.path.exists(img_name):
                        
                        image = io.imread(img_name)
                        if transform:
                            image = transform(image)
                        image = image.to(device)
                        
                        # image of article to models
                        for i in range(len(models)):
                            end = group_sizes_cumm[i]
                            start = end - group_sizes[i]
                            print("recom:",recommandation.device)
                            recommandation[start:end] += models[i](image.unsqueeze(0)).squeeze(0)
                            
                recommandation.to("cpu")
                indices = recommandation.topk(num_reccom).indices
                
                articles = ""
                for i in range(num_reccom):
                    index = indices[i].item()
                    j = 0
                    while(index>=group_sizes_cumm[j]):
                        j += 1
                    # image in group j
                    if(j>0):
                        index -= group_sizes_cumm[j-1]
                    # image index in group j is index
                    articles += group2id[(j,index)]+ " "
                articles = articles[:-1]
                
        if k % 1000 == 1000-1:
            print('%d customers predicted' %(k + 1))
            print("time:",time.time()-begin_time)
        submission.writerow([customer,articles])
                     
    submission_file.close()
    print("number of new customer", num_new_customer, "total customer", len(customers_list))

if __name__ == '__main__':
    start_time = time.time()
    print('Is cuda available?', torch.cuda.is_available())
    
    images_dir = '/home/Biao/data/images__all/'
    transactions_dir = '/home/Biao/data/transactions_train.csv'
    transactions_dir_train = '/home/Biao/data/transactions_train_train1month.csv'
    transactions_dir_valid = '/home/Biao/data/transactions_train_test1week.csv'
    articles_dir = '/home/Biao/data/articles_1month.csv'
    customers_dir = '/home/Biao/data/customers.csv'
    predictions_dir = '/home/Biao/data/submission_1week.csv'
    graph_dir = '/home/Biao/H-M-Personalized-Fashion-Recommendations/data/'
    
    batch_size = 64
    max_epoch = 10
    rate = 1e-3
    train_period = 10
    num_recomm = 12
    
    #num_articles = len(os.listdir(images_dir)) #105100
    myTransform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])
    
    with open("group2id.pkl", "rb") as f:
        group2id = pickle.load(f)
    with open("id2group.pkl", "rb") as f:
        id2group = pickle.load(f)
    with open("group_sizes.pkl", "rb") as f:
        group_sizes = pickle.load(f)

    models = []
    for i in range(len(group_sizes)):
        model = Model(group_length=group_sizes[i])
        model.load_state_dict(torch.load(graph_dir+"model"+str(i)+".pt"))
        models.append(model)
    
    # change in tr_dir 
    predictions(models,id2group=id2group,group2id=group2id,group_sizes=group_sizes,\
            num_reccom=num_recomm,tr_dir_train=transactions_dir_train,tr_dir_valid=transactions_dir_valid,\
            pred_dir=predictions_dir,images_dir=images_dir,transform=myTransform)
    
    print("making final predictions for approximately the optimal value of epoch : --- %s seconds ---" % (time.time() - start_time))

    map12 = score(tr_dir=transactions_dir_valid,pred_dir=predictions_dir,num_recomm=num_recomm)
    print("calculating map@12 : --- %s seconds ---" % (time.time() - start_time))
    print("map@12:", map12)