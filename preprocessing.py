import pandas as pd

def transactionSplit(dataAdress, date_train_start, date_train_end):
    df = pd.read_csv(dataAdress, usecols=['t_dat','customer_id','article_id'], dtype={'article_id':str})
    index_start = df[df['t_dat'] == date_train_start].index.tolist()[0]
    index_end = df[df['t_dat'] == date_train_end].index.tolist()[0]
    df_train = df.iloc[index_start:index_end]
    file_name = dataAdress[:-4] + '_train1month.csv'
    df_train.to_csv(file_name)
    df_test = df.iloc[index_end:]
    file_name = dataAdress[:-4] + '_test1week.csv'
    df_test.to_csv(file_name)

def articleSplit(tr_train_dir,ar_dir):
    tr_df = pd.read_csv(tr_train_dir, usecols=['t_dat','customer_id','article_id'], dtype={'article_id':str})
    article_ids = set(tr_df['article_id'].unique())
    
    ar_df = pd.read_csv(ar_dir, dtype={'article_id':str})
    L_drop = []
    for i,row in ar_df.iterrows():
        if(row['article_id'] not in article_ids):
            L_drop.append(i)
            
    ar_df = ar_df.drop(L_drop)
    ar_df.to_csv(ar_dir[:-4] + '_1month.csv')

dataAdress = "~/data/transactions_train.csv"
transactionSplit(dataAdress, '2020-08-16', '2020-09-16')
articleSplit(dataAdress[:-4] + '_train1month.csv', "~/data/articles.csv")