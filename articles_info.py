import pandas as pd

# df = pd.read_csv('/home/aymen/data/articles.csv',dtype={'article_id':str})
df = pd.read_csv('./data/articles.csv',usecols=['article_id','product_group_name','product_type_name'])
tr_df = pd.read_csv('./data/transactions_train.csv',usecols=['article_id'])

n_article = df['article_id'].count()
print('number of article is', n_article)

print('number of article in transactions_train is', len(tr_df['article_id'].unique()))

print(df['product_group_name'].value_counts())

print('==========Garment Upper body==========')
subdf = df.loc[df['product_group_name'] == 'Garment Upper body']
print(subdf['product_type_name'].value_counts())

print('==========Garment Lower body==========')
subdf = df.loc[df['product_group_name'] == 'Garment Lower body']
print(subdf['product_type_name'].value_counts())

