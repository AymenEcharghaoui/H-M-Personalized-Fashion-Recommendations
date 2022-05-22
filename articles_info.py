import pandas as pd

# df = pd.read_csv('/home/aymen/data/articles.csv',dtype={'article_id':str})
df = pd.read_csv('./data/articles.csv',usecols=['article_id','product_group_name','product_type_name'],dtype={'article_id':str})
tr_df = pd.read_csv('./data/transactions_train.csv',usecols=['article_id'],dtype={'article_id':str})

n_article = len(df['article_id'].unique())
print('number of article is', n_article)

print('number of article in transactions_train is', len(tr_df['article_id'].unique()))

# print(df['product_group_name'].value_counts())

# print('==========Garment Upper body==========')
# subdf = df.loc[df['product_group_name'] == 'Garment Upper body']
# print(subdf['product_type_name'].value_counts())

# print('==========Garment Lower body==========')
# subdf = df.loc[df['product_group_name'] == 'Garment Lower body']
# print(subdf['product_type_name'].value_counts())

im = open("./ids.txt","r")
im_set = set()
for line in im:
    im_set.add(line[:-5])
im.close()

print('number of article in images__all is', len(im_set))

ar_set = set(df['article_id'].unique())
tr_set = set(tr_df['article_id'].unique())

print('articles in articles.csv not in transactions_train.csv:', len(ar_set - tr_set))

print('articles in transactions_train.csv not in articles.csv:', len(tr_set - ar_set))

print('articles in articles.csv not in images__all:', len(ar_set - im_set))

print('articles in images__all not in articles.csv:', len(im_set - ar_set))

print('articles in images__all not in transactions_train.csv:', len(im_set - tr_set))

print('articles in transactions_train.csv not in aimages__all:', len(tr_set - im_set))



