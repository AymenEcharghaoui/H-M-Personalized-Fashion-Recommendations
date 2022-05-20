import pandas as pd
import os
import shutil


df = pd.read_csv('/home/aymen/data/articles.csv',dtype={'article_id':str,'product_group_name':str})

groups = list(df['product_group_name'].unique())
print((groups[0][0]))


for i in range(len(groups)):
    if(groups[i]  != 'Underwear/nightwear'):
        groups[i] = groups[i].replace("&","and").replace(" ","_")
        if(not os.path.exists('/home/aymen/data/'+groups[i])):
            os.mkdir('/home/aymen/data/'+groups[i])

for i,row in df.iterrows():
    if row['product_group_name']!= 'Underwear/nightwear':
        row['product_group_name'] = row['product_group_name'].replace("&","and").replace(" ","_")
        if(os.path.exists('/home/aymen/data/images__all/'+str(row['article_id'])+'.jpg')):
            os.replace('/home/aymen/data/images__all/'+str(row['article_id'])+'.jpg','/home/aymen/data/'+row['product_group_name']+'/'+row['article_id']+'.jpg')
    else:
        if(os.path.exists('/home/aymen/data/images__all/'+str(row['article_id'])+'.jpg')):
            shutil.copy('/home/aymen/data/images__all/'+str(row['article_id'])+'.jpg','/home/aymen/data/Underwear/'+row['article_id']+'.jpg')
            os.replace('/home/aymen/data/images__all/'+str(row['article_id'])+'.jpg','/home/aymen/data/Nightwear/'+row['article_id']+'.jpg')
