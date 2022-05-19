import pandas as pd
import os


df = pd.read_csv('/home/aymen/data/articles.csv',dtype={'article_id':str})

groups = list(df['product_group_name'].unique())
print(groups)

for group in groups:
    #os.mkdir('/home/aymen/data/'+group[1:-1])
    print(group[1:-1])
'''
for i,row in df.iterrows():
    os.replace('/home/aymen/data/images_all/'+str(row['article_id'])+'.jpg','/home/aymen/data/'+row['product_group_name']+'/'+row['article_id']+'.jpg')
'''