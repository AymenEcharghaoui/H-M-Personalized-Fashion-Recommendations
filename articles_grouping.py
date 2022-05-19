import pandas as pd
import os


df = pd.read_csv('/home/aymen/data/articles.csv',dtype={'article_id':str,'product_group_name':str})

groups = list(df['product_group_name'].unique())
print((groups[0][0]))


for i in range(len(groups)):
    if(groups[i]  != 'Underwear/nightwear'):
        groups[i] = groups[i].replace("'","").replace(" ","_")
        os.mkdir('/home/aymen/data/'+groups[i])

for i,row in df.iterrows():
    if row['product_group_name']!= 'Underwear/nightwear':
        row['product_group_name'] = row['product_group_name'].replace("'","").replace(" ","_")
        os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/'+row['product_group_name']+'/'+row['article_id']+'.jpg')
    else:
        os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/Underwear/'+row['article_id']+'.jpg')
        os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/Nightwear/'+row['article_id']+'.jpg')
