import pandas as pd
import datetime

dir_tr = ""
transactions = pd.read_csv(dir_tr,dtype={'article_id':str})
transactions['t_dat'] = transactions['t_dat'].to_datetime()

print("all transactions from {} to {}".format(transactions['t_dat'].min(),transactions['t_dat'].max()))

transactions_train = transactions.loc[(transactions['t_dat']<datetime.datetime(2020,9,16) and transactions['t_dat']>= datetime.datetime(2020,9,8))]

transactions_val = transactions.loc[(transactions['t_dat']>=datetime.datetime(2020,9,16))]

items = transactions_train.groupby('customers_id')['article_id'].apply(list)

print(items.head())
