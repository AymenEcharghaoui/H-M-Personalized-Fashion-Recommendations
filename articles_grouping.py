import pandas as pd
import os


df = pd.read_csv('/home/aymen/data/articles.csv')

groups = df['product_group_name'].unique()

for group in groups:
    os.mkdir('/home/aymen/data/'+group)

for i,row in df.iterrows():
    os.replace('/home/aymen/data/images_all/'+row['article_id']+'.jpg','/home/aymen/data/'+row['product_group_name']+'/'+row['article_id']+'.jpg')