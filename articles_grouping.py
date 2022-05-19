import pandas as pd
import os


df = pd.read_csv('/home/aymen/data/articles.csv',dtype={'article_id':str,'product_group_name':str})

groups = list(df['product_group_name'].unique())
print(type(groups[0]))

'''
for group in groups:
    if(group  != 'Underwear/nightwear'):
        os.mkdir('/home/aymen/data/'+group)

for i,row in df.iterrows():
    if row['product_group_name']!= 'Underwear/nightwear':
        os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/'+row['product_group_name']+'/'+row['article_id']+'.jpg')
    else:
        os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/Underwear/'+row['article_id']+'.jpg')
        os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/Nightwear/'+row['article_id']+'.jpg')
'''