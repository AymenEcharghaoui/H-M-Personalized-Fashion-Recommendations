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
        
def loadModel(model_dir):
    with open(model_dir, "rb") as f:
        model = pickle.load(f)
    return model


class Model(torch.nn.Module):

    def __init__(self,group_length,activation = torch.nn.ReLU()) :
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,4,kernel_size=3,stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        self.conv3 = torch.nn.Conv2d(4,5,kernel_size=3,stride=1,padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        self.dense5 = torch.nn.Linear(5*14*14,int(group_length/2))
        self.batchNorm = torch.nn.BatchNorm1d(int(group_length/2))
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
        z = self.batchNorm(z)
        z = self.activation(z)
        z = self.dense6(z)

        return z

def trainer_all(train_datasets,models,batch_size,loss_fn,max_epoch,rate,train_period,id2group,group2id,group_sizes,\
    tr_train_dir,cust_dir,pred_dir,images_dir,graph_dir,num_reccom,transform,tr_valid_dir=None):

    training_generators = [DataLoader(train_datasets[i], batch_size = batch_size,shuffle = True, num_workers = 4) for i in range(len(group_sizes))]
    optimizers = [torch.optim.Adam(params=models[i].parameters(),lr=rate,weight_decay=1e-4) for i in range(len(group_sizes))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # size epoch * #Groups
    train_loss = np.zeros((max_epoch,len(group_sizes)))
    # size epoch
    train_map = []
    valid_map = []

    i = 0
    while(i < max_epoch):
        for j in range(len(group_sizes)):
            running_loss = 0.0
            total = 0
            models[j].to(device)
            models[j].train()
            for k,sample_batched in enumerate(training_generators[j]):
                x,y = sample_batched
                x = x.to(device)
                y = y.to(device)
                optimizers[j].zero_grad()
                y_pred = models[j](x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizers[j].step()
                
                total += y.size(0)
                running_loss += loss.item() * y.size(0)
                
                if k % train_period == train_period-1:
                    print('epoch:%d, period:%d running loss: %.3f' %(i , k , loss.item()))

            models[j].to(torch.device('cpu'))
            torch.cuda.empty_cache()

            running_loss = running_loss/total
            print('epoch:%d average loss: %.3f' %(i , running_loss))
            train_loss[i][j] = running_loss

            torch.save(models[j].state_dict(), graph_dir + 'model'+str(j)+'.pt')

        if(tr_valid_dir):
            pred_train_dir = pred_dir[:-4]+ '_train'+str(i)+ '.csv'
            predictions(models,id2group,group2id,group_sizes,tr_train_dir,cust_dir,pred_train_dir,images_dir,num_reccom=num_reccom,transform=transform)
            train_map.append(score(tr_train_dir,pred_train_dir,num_recomm))
            print('train_map %.4f for epoch: %d' %(train_map[-1],i))

            pred_valid_dir = pred_dir[:-4]+ '_valid'+str(i) + '.csv'
            predictions(models,id2group,group2id,group_sizes,tr_valid_dir,cust_dir,pred_valid_dir,images_dir,num_reccom=num_reccom,transform=transform)
            valid_map.append(score(tr_valid_dir,pred_valid_dir,num_recomm))
            print('test_map %.4f for epoch: %d' %(valid_map[-1],i))

            if(i>=2 and valid_map[-1]<=valid_map[-2] and valid_map[-2]<=valid_map[-3]):
                break
        i += 1
        
    return i,train_loss,train_map,valid_map

def score(tr_dir,pred_dir,num_recomm=12):
    """
    return MAP@12 over the test dataset
    Args : 
        tr_dir (string): directory of transactions_train_valid.csv
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
        customer_id = row[0]
        if(customer_id in transactions):
            count_customers += 1
            # going through all customers
            average_precision = 0
            reccomandations = row[1].split()
            relevant = 0
            for i in range(num_recomm):
                if(reccomandations[i] in transactions[customer_id]):
                    relevant += 1
                    average_precision += relevant/(i+1)
            average_precision /= min(12,len(transactions[customer_id]))
            map12 += average_precision
    map12 /= count_customers
    return map12

def predictions(models,id2group,group2id,group_sizes,tr_dir,cust_dir,pred_dir,images_dir,num_reccom=12,transform=None) :
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
    for i in range(len(models)):
        models[i].eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group_sizes_cumm = []
    start = 0
    for size in group_sizes:
        start += size
        group_sizes_cumm.append(start)

    num_articles = group_sizes_cumm[-1]

    recommandations = {}


    '''
    # all customers : there are new customers in customers.csv
    customers = pd.read_csv(cust_dir,dtype={'article_id':str})
    for i,row in customers.iterrows():
        recommandations[row['customer_id']] = torch.zeros(num_articles,dtype=torch.float32)
        is_active[row['customer_id']] = False
    '''

    # looking at only a subset of customers
    # making recommandations based on previous transactions
    transactions = pd.read_csv(tr_dir,dtype={'article_id':str})

    seen = set()
    for i,row in transactions.iterrows():
        if(not(row['customer_id'] in seen)):
            recommandations[row['customer_id']] = torch.zeros(num_articles,dtype=torch.float32)
            seen.add(row['customer_id'])

    for i,row in transactions.iterrows():

        assert row['customer_id'] in recommandations

        image_id = row['article_id']
        group_index = id2group[image_id][0]
        img_name = os.path.join(images_dir,'0'+str(image_id)+'.jpg')
        if(os.path.exists(img_name)):
            image = io.imread(img_name)
            if transform:
                image = transform(image)
            image = image.to(device)
            #start = sum(group_sizes[:id2group[image_id][0]])
            #end = start + group_sizes[id2group[image_id][0]]
            end = group_sizes_cumm[id2group[image_id][0]]
            start = end - group_sizes[id2group[image_id][0]]

            models[group_index].to(device)
            recommandations[row['customer_id']][start:end] += models[group_index](image.unsqueeze(0)).squeeze(0).to('cpu')
            models[group_index].to(torch.device('cpu'))

    submission_file = open(pred_dir,'w',newline='')
    # no worries of a second execution : we overwrite what's already existing in the submission file

    submission = csv.writer(submission_file,delimiter=',')

    submission.writerow(['customer_id','prediction'])

    #for i,row in customers.iterrows():
    for customer in recommandations:
        line = [customer]
        articles = ""
        reccs = recommandations[customer]

        if(not(row['customer_id'] in seen)):
            '''
            # new customer : generate num_reccom random articles
            for _ in range(num_reccom-1):
                articles += os.listdir(images_dir)[random.randint(0,num_articles-1)][:-4]+" "
            articles += os.listdir(images_dir)[random.randint(0,num_articles-1)][:-4]
            '''
            continue

        else:
            indices = reccs.topk(num_reccom).indices
            '''
            for i in range(num_reccom-1):
                articles += os.listdir(images_dir)[indices[i]][:-4]+ " "
            articles += os.listdir(images_dir)[indices[num_reccom-1]][:-4]
            '''
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
            articles  = articles[:-1]

        line.append(articles)
        submission.writerow(line,)

    submission_file.close()


def lossPlot(loss,dir,i,epoch):
    # per group
    plt.figure()
    plt.plot(loss[:epoch+1],label = "loss for training set")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dir+"loss_curves_"+str(i)+".png")
    
    file = open(dir+"loss"+str(i)+".txt", "w")
    for element in loss:
        file.write(str(element) + "\n")
    file.close()

def mapPlot(train_map,valid_map,graph_dir,epoch):
    # per epoch for all groups
    plt.figure()
    plt.plot(train_map[:epoch+1],label = "map@12 for training set")
    plt.plot(valid_map[:epoch+1],label = "map@12 for validating set")
    plt.xlabel("Epoch")
    plt.ylabel("map@12")
    plt.legend()
    plt.savefig(graph_dir+"score_curves_.png")
    
    file = open(graph_dir+"score.txt", "w")
    for element in train_map:
        file.write(str(element) + "\n")
    for element in valid_map:
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
    transactions_dir_train = '/home/aymen/data/transactions_train_train1month.csv'
    transactions_dir_valid = '/home/aymen/data/transactions_train_test1week.csv'
    articles_dir = '/home/aymen/data/articles_1month.csv'
    customers_dir = '/home/aymen/data/customers.csv'
    predictions_dir = '/home/aymen/data/submission.csv'
    graph_dir = '/home/aymen/data/'
    
    
    '''
    images_dir = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/images/images_test/'
    transactions_dir = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/transactions_train_20.csv'
    transactions_dir_train = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/transactions_train_20.csv'
    transactions_dir_valid = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/transactions_train_20.csv' #TO DO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    articles_dir = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/articles.csv'
    customers_dir = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/customers_20.csv'
    predictions_dir = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/submission.csv'
    graph_dir = '/home/echarghaoui/github_Aymen/H-M-Personalized-Fashion-Recommendations/data/'
    '''
    
    batch_size = 64
    max_epoch = 10
    rate = 1e-3
    train_period = 10
    num_recomm = 12
    
    #num_articles = len(os.listdir(images_dir)) #105100
    myTransform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])
    
    # based on all 105543 articles
    #group2id,id2group,group_sizes = createArticlesDic(articles_dir=articles_dir)

    '''
    #just for test
    images_set = set([image[:-4] for image in os.listdir(images_dir)])
    group2id = {}
    id2group = {}
    i = 0
    for image in images_set:
        if(i<10):
            id2group[image] = (0,i)
            group2id[(0,i)] = image
            i += 1
        else:
            id2group[image] = (1,i-10)
            group2id[(1,i-10)] = image
            i += 1
    group_sizes = [10,10]
    '''

    '''
    train_datasets,relevant,id_relevant = createDataset(images_dir=images_dir,id2group=id2group,group_sizes=group_sizes,\
        transactions_dir=transactions_dir_train, transform = myTransform)
    
    print("creating datasets : --- %s seconds ---" % (time.time() - start_time))
    '''
    #saveDatasets(group2id,id2group,group_sizes,relevant,id_relevant)

    
    group2id,id2group,group_sizes,train_datasets = loadDatasets(images_dir=images_dir,transform=myTransform)
    '''
    models = []
    for i in range(len(group_sizes)):
        model = Model(group_length=group_sizes[i])
        model.apply(init_weights_xavier_uniform)
        models.append(model)

    opt_epoch,train_loss,train_map,valid_map = trainer_all(train_datasets,models,batch_size,torch.nn.BCEWithLogitsLoss(), \
            max_epoch=max_epoch,rate=rate, train_period=train_period,id2group=id2group,group2id=group2id, \
            group_sizes=group_sizes,tr_train_dir=transactions_dir_train,tr_valid_dir=transactions_dir_valid,\
            cust_dir=customers_dir,pred_dir=predictions_dir,images_dir=images_dir,graph_dir=graph_dir,\
            num_reccom=num_recomm,transform=myTransform)

    assert len(train_map)==opt_epoch+1
    print("training of all models"+": --- %s seconds ---" % (time.time() - start_time))
    

    for i in range(len(group_sizes)):
        # saving all 10 models per group 
        
        #model_submit_dir = './data/model'+str(i)+'.pt'
        #torch.save(models[i].state_dict(), model_submit_dir)
    
        lossPlot(train_loss[:,i],graph_dir,i,opt_epoch)
    
    mapPlot(train_map,valid_map,graph_dir,opt_epoch)
    '''

    models = []
    for i in range(len(group_sizes)):
        model = Model(group_length=group_sizes[i])
        model.load_state_dict(torch.load("model"+str(i)+".pt"))
        model.eval()
        models.append(model)
    
    # change in tr_dir 
    predictions(models,id2group=id2group,group2id=group2id,group_sizes=group_sizes,num_reccom=num_recomm,tr_dir=transactions_dir_train,cust_dir=customers_dir,pred_dir=predictions_dir,images_dir=images_dir,transform=myTransform)
    
    print("making final predictions for approximately the optimal value of epoch : --- %s seconds ---" % (time.time() - start_time))

    print(score(tr_dir=transactions_dir_valid,pred_dir=predictions_dir,num_recomm=num_recomm))

    '''
    to load models : 
    model = Model(num_articles=num_articles)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    '''