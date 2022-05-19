import pandas as pd
import os


df = pd.read_csv('/home/aymen/data/articles.csv',dtype={'article_id':str})

groups = list(df['product_group_name'].unique())
print(groups)

os.mkdir('/home/aymen/data/Underwear')
for group in groups:
    if(group  != 'Underwear/nightwear'):
        os.mkdir('/home/aymen/data/'+group)

for i,row in df.iterrows():
    prod_type = row['product_group_name'] if row['product_group_name']!= 'Underwear/nightwear' else 'Underwear'
    os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/'+prod_type+'/'+row['article_id']+'.jpg')
