# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:46:51 2022

@author: Biao
"""

import pandas as pd

def dataSplit(dataAdress, usecols, date):
    df = pd.read_csv(dataAdress, usecols=usecols)
    index = df[df['t_dat'] == date].index.tolist()[0]
    df_train = df.iloc[:index]
    file_name = dataAdress[:-4] + '_train.csv'
    df_train.to_csv(file_name)
    df_test = df.iloc[index:]
    file_name = dataAdress[:-4] + '_test.csv'
    df_test.to_csv(file_name)
    
    return (df_train, df_test)
    
def dataHead(dataAdress, usecols, head):
    df = pd.read_csv(dataAdress, usecols=usecols, nrows=head)
    file_name = dataAdress[:-4] + '_' + str(head) + '.csv'
    df.to_csv(file_name)
    
    return df
    
if __name__ == '__main__':
    dataAdress = "~/data/transactions_train.csv"
    # df = pd.read_csv(dataAdress)
    # print(df.iloc[-1])
    '''
    t_dat                                                      2020-09-22
    customer_id         fffef3b6b73545df065b521e19f64bf6fe93bfd450ab20...
    article_id                                                  898573003
    price                                                        0.033881
    sales_channel_id                                                    2
    Name: 31788323, dtype: object
    '''
    
    usecols=['t_dat','customer_id','article_id']
    dataSplit(dataAdress, usecols, '2020-09-16')
    
    # dataHead(dataAdress, usecols, head=10)
    
    # dataHead("./data/customers.csv", None, 10)
    