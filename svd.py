import pandas as pd
import datetime

transactions_dir = '/home/aymen/data/transactions_train.csv'
transactions = pd.read_csv(transactions_dir,dtype={'article_id':str})
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

print("all transactions from {} to {}".format(transactions['t_dat'].min(),transactions['t_dat'].max()))

transactions_train = transactions.loc[(transactions['t_dat']<datetime.datetime(2020,9,16) & transactions['t_dat']>= datetime.datetime(2020,9,8))]

transactions_val = transactions.loc[(transactions['t_dat']>=datetime.datetime(2020,9,16))]

items = transactions_train.groupby('customers_id')['article_id'].apply(list)

print(items.head())
