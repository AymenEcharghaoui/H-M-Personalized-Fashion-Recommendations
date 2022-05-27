import os
import torch
from skimage import io, transform
import pandas as pd
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import time
import pickle
import torch.utils.model_zoo as model_zoo


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
        
    return (group2id,id2group,group_sizes,datasets,relevant,id_relevant)

def creatTestDataset(images_dir, group2id,id2group,group_sizes, transactions_dir, transform=None):
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
        Directory of the transactions of testset csv file.
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
    
    # creat a set of all articles_id with image
    images_set = set([image[:-4] for image in os.listdir(images_dir)])
    
    # creat relevant dir {article: a set of all articles(1d2group[article]) relevant to it}
    relevant = {}
    id_relevant = []
    for customer in transactions:
        for p_article in transactions[customer]:
            if p_article in images_set:
                for article in transactions[customer]:
                    if p_article != article and article in id2group:
                        if p_article not in relevant:
                            relevant[p_article] = {id2group[article]}
                            id_relevant.append(p_article)
                        else:
                            relevant[p_article].add(id2group[article])
    
    # creat datasets
    datasets = [] #list of datasets
    for i in range(len(group_sizes)):
        datasets.append(ArticlesDataset(i,images_dir,group_sizes[i],relevant,id_relevant,transform=transform))
        
    return (datasets,relevant,id_relevant)

def saveDatasets(group2id,id2group,group_sizes,relevant,id_relevant,isTrain = True):
    if isTrain:
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
    else:
        with open("test_relevant.pkl", "wb") as f:
            pickle.dump(relevant,f,protocol=pickle.HIGHEST_PROTOCOL)
        with open("test_id_relevant.pkl", "wb") as f:
            pickle.dump(id_relevant,f,protocol=pickle.HIGHEST_PROTOCOL)
        
def loadDatasets(images_dir, transform=None, isTrain = True):
    with open("group2id.pkl", "rb") as f:
        group2id = pickle.load(f)
    with open("id2group.pkl", "rb") as f:
        id2group = pickle.load(f)
    with open("group_sizes.pkl", "rb") as f:
        group_sizes = pickle.load(f)
    if isTrain:
        with open("relevant.pkl", "rb") as f:
            relevant = pickle.load(f)
        with open("id_relevant.pkl", "rb") as f:
            id_relevant = pickle.load(f)
    else:
        with open("test_relevant.pkl", "rb") as f:
            relevant = pickle.load(f)
        with open("test_id_relevant.pkl", "rb") as f:
            id_relevant = pickle.load(f)
    
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


def trainer(train_dataset,test_dataset,model,loss_fn,batch_size,epoch,rate,train_period) :
    training_generator = DataLoader(train_dataset, batch_size = batch_size,shuffle = True, num_workers = 5)
    
    test_subset = torch.utils.data.Subset(test_dataset, list(range(2560)))
    
    begin_time = time.time()
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=rate,weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loss = []
    test_loss = []
    
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
            running_loss += loss.detach() * y.size(0)
            
            if j % train_period == train_period-1:
                print('epoch:%d, period:%d running loss: %.5f' %(i + 1, j + 1, loss.detach()))
                print("time:",time.time()-begin_time)
        
        running_loss = running_loss/total
        print('epoch:%d average loss: %.5f' %(i + 1, running_loss))
        print("time:",time.time()-begin_time)
        train_loss.append(running_loss)
        
        average_test_loss = testLoss(test_subset,model,loss_fn,batch_size)
        print('epoch:%d test loss: %.5f' %(i + 1, average_test_loss))
        print("time:",time.time()-begin_time)
        test_loss.append(test_loss)
        
    return train_loss, test_loss
    
def testLoss(test_dataset,model,loss_fn,batch_size):
    model.eval()
    
    test_generator = DataLoader(test_dataset, batch_size = batch_size,shuffle = True, num_workers = 5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for j, sample_batched in enumerate(test_generator):
            x,y = sample_batched
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            total += y.size(0)
            running_loss += loss.detach() * y.size(0)
    
    running_loss = running_loss/total
    
    model.train()
    return running_loss
    
    
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
        
class AlexNet(torch.nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,11,stride=4,padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Conv2d(64,192,5,stride=1,padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Conv2d(192,384,3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384,256,3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256,256,3,stride=1,padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view([x.size()[0],256 * 6 * 6])
        x = self.classifier(x)
        return x

model_urls = {
'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def alexnet_classifier(num_classes):
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(256 * 6 * 6, int(num_classes/2) ),
        torch.nn.BatchNorm1d(int(num_classes/2)),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(int(num_classes/2), num_classes),
    )
    return classifier

def alexnet(num_classes, pretrained=False, **kwargs):
    """AlexNet model architecture from the "One weird trick..." paper.
    Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        for p in model.features.parameters():
            p.requires_grad=False
    classifier = alexnet_classifier(num_classes)
    model.classifier = classifier
    return model


if __name__ == '__main__':
    start_time = time.time()
    print('Is cuda available?', torch.cuda.is_available())
    
    images_dir = '/home/Biao/data/images__all/'
    transactions_dir_train = '/home/Biao/data/transactions_train_train1month.csv'
    transactions_dir_test = '/home/Biao/data/transactions_train_test1week.csv'
    articles_dir = '/home/Biao/data/articles_1month.csv'
    loss_dir = '/home/Biao/data/'
    '''
    images_dir = './data/images/images_test/'
    transactions_dir_train = './data/transactions_train_train1month.csv'
    transactions_dir_test = './data/transactions_train_test1week.csv'
    articles_dir = './data/articles_1month.csv'
    loss_dir = './data/'
    '''
    
    batch_size = 512
    epoch = 5
    rate = 1e-3
    train_period = 5
    num_recomm = 12
    
    myTransform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    
    # (group2id,id2group,group_sizes,datasets,relevant,id_relevant) = creatDataset(images_dir, articles_dir, transactions_dir_train, transform = myTransform)
    # saveDatasets(group2id, id2group, group_sizes, relevant, id_relevant)
    (group2id,id2group,group_sizes,datasets) = loadDatasets(images_dir,transform=myTransform)
    print("creating train dataset : --- %s seconds ---" % (time.time() - start_time))
    print("number of train dataset: ", len(datasets[0]))
    
    # (test_datasets,test_relevant,test_id_relevant) = creatTestDataset(images_dir, group2id,id2group,group_sizes, transactions_dir_test, transform = myTransform)
    # saveDatasets(group2id, id2group, group_sizes, test_relevant, test_id_relevant, isTrain=False)
    (group2id,id2group,group_sizes,test_datasets) = loadDatasets(images_dir,transform=myTransform, isTrain=False)
    print("creating test dataset : --- %s seconds ---" % (time.time() - start_time))
    print("number of test dataset: ", len(test_datasets[0]))

    # models = []
    for i in range(0, 1):

        model_submit_dir = './data/second_try_model'+str(i)+'.pt'

        dataset = datasets[i]
        test_dataset = test_datasets[i]
        print("dataset "+str(i)+": --- %s seconds ---" % (time.time() - start_time))
        
        model = alexnet(num_classes=group_sizes[i], pretrained=True)
        model.apply(init_weights_xavier_uniform)
        if(torch.cuda.is_available()):
            model.cuda()
        train_loss,test_loss = trainer(dataset,test_dataset, model,torch.nn.BCEWithLogitsLoss(),batch_size=batch_size,epoch=epoch,rate=rate, train_period=train_period)
        torch.save(model.state_dict(), model_submit_dir)
        
        with open("second_try_train_loss"+str(i)+".pkl", "wb") as f:
            pickle.dump(train_loss,f,protocol=pickle.HIGHEST_PROTOCOL)

        with open("second_try_test_loss"+str(i)+".pkl", "wb") as f:
            pickle.dump(test_loss,f,protocol=pickle.HIGHEST_PROTOCOL)
        # lossPlot(train_loss,loss_dir,i)
        print("training "+str(i)+": --- %s seconds ---" % (time.time() - start_time))
        model.to(torch.device('cpu'))
        torch.cuda.empty_cache()
        # models.append(model)

# ======Biao's test=========
# (group2id,id2group,group_sizes,datasets) = creatDataset('./data/images/images_test/', './data/articles.csv', './data/transactions_train_10.csv', transform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()]))
# print(datasets[0][0][0][1][11][11])

# (group2id,id2group,group_sizes) = creatArticlesDic('./data/articles_1month.csv')
# print(group_sizes)
# print(id2group)
# import numpy as np
# (group2id,id2group,group_sizes,datasets) = loadDatasets("")
# print(np.mean(group_sizes))
