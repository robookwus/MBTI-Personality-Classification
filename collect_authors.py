# !pip install pmaw
import praw
import pandas as pd
from praw.models import MoreComments
import os

from pmaw import PushshiftAPI
import datetime as dt


reddit = praw.Reddit(
    client_id="DRsyDcTSsKxZsPw4T1vt_g",
    client_secret="NO8JknCAyXr4UGmRuMEr7nWd4eEq8g", 
    password="KwLFJEHigeiNM9K",
    user_agent="testscript by u/mbti_scrap", 
    username="mbti_scrap",
)

################################################################

# approach with pmaw
# collect all comments and meta data
from pmaw import PushshiftAPI
import datetime as dt
years = [2018, 2019, 2020, 2021, 2022]
api = PushshiftAPI()
year=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for month in year:
    start_epoch=  int(dt.datetime(2022, month, 1).timestamp()) 
    end_epoch=  int(dt.datetime(2022, month, 28).timestamp()) # month is just day 1 - 28 for ease of implementation
    print(start_epoch)
    comments = api.search_comments(subreddit="mbti", #retrieve all comments of month
                                   limit=None,
                                   after=start_epoch,
                                   before=end_epoch)
    comments_df = pd.DataFrame(comments)
    comments_df.to_csv(f'./mbti_all_comments_{month}_22.csv', header=True, index=False, columns=list(comments_df.axes[1]))
    print(f'Total: Retrieved {len(comments)} total comments from Pushshift in month {month}.2022.')
    
    print(comments_df.head(5)) # preview the comments data


################################################################


years = [18, 19, 20, 21, 22]
year=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def concat_authors():
    """
    reads all authers and their flairs of the scraped comments csv files and puts them in one dataframe. 
    returns authors + labels dataframe.
    """
    
    authors_list = []
    labels_list = []
    for year_count in years:
        for month in year:
            month_df = pd.read_csv(f'./mbti_all_comments_{month}_{year_count}.csv', usecols = ['author','author_flair_text'])
            authors = month_df['author'].tolist()
            labels =  month_df['author_flair_text'].tolist()
            authors_list += authors
            labels_list += labels
    author_labels_tup = list(zip(authors_list, labels_list)) #make tuple
    total_authors_df = pd.DataFrame(author_labels_tup, columns=['authors','labels'])
    print(total_authors_df.head(10)) # preview the authors data
    total_authors_df.to_csv(f'./mbti_authors_total.csv', header=True, index=False, columns=list(total_authors_df.axes[1]))
    return total_authors_df
concat_authors()


def remove_duplicates():
    """
    clean up authors: remove without labels
    """
    total_authors_df = pd.read_csv('mbti_authors_total.csv')
    print(f"Total number of collected authors: {len(total_authors_df)}")
    total_authors_df = total_authors_df.dropna(subset=['labels']) # remove authors without labels (deleted users have no labels)
    print(f"Number of collected authors with label: {len(total_authors_df)}")
    total_authors_df = total_authors_df.drop_duplicates(subset=['authors'])# remove duplicate authors
    print(f"Number of collected authors without duplicates: {len(total_authors_df)}")
    total_authors_df.to_csv(f'./mbti_authors_cleaned.csv', header=True, index=False, columns=list(total_authors_df.axes[1]))
    return total_authors_df
remove_duplicates()

import matplotlib.pyplot as plt
authors_df = remove_duplicates()
(
    
    authors_df['labels']
    .value_counts()
    .plot.bar()
    #.savefig('mbti_authors_dist.png')
)    


# approach with pmaw
# collect all comments and meta data
# for sparsely represented subreddits
"""
Collect all comments from bottom 8 personality types.
Stores them in csv in the respective folders.
"""
from pmaw import PushshiftAPI
import datetime as dt
years = [2018, 2019, 2020, 2021, 2022]
class_subreddits = ['ESFJ', 'ESTJ', 'ISFJ', 'ESFP', 'ISTJ', 'ENFJ', 'ISFP', 'ESTP']
api = PushshiftAPI()
#year=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for class_subreddit in class_subreddits:
    for year in years:
        start_epoch=  int(dt.datetime(year, 1, 1).timestamp()) 
        end_epoch=  int(dt.datetime(year, 12, 31).timestamp()) # month is just day 1 - 28 for ease of implementation
        print(start_epoch)
        comments = api.search_comments(subreddit=class_subreddit, #retrieve all comments of month
                                       limit=None,
                                       after=start_epoch,
                                       before=end_epoch)
        comments_df = pd.DataFrame(comments)
        comments_df.to_csv(f'./{class_subreddit}/{class_subreddit}_all_comments_{year}.csv', header=True, index=False, columns=list(comments_df.axes[1]))
        print(f'Total: Retrieved {len(comments_df)} total comments from Pushshift in year {year}.')

        print(comments_df.head(5)) # preview the comments data


years = [18, 19, 20, 21, 22]
class_subreddits = ['ESFJ', 'ESTJ', 'ISFJ', 'ESFP', 'ISTJ', 'ENFJ', 'ISFP', 'ESTP']

def concat_additional_authors():
    """
    reads all authers and their flairs of the sparse subreddits from the scraped comments csv files and puts them in one dataframe. 
    returns authors + labels of sparse subreddits dataframe.
    """
    
    authors_list = []
    labels_list = []
    for class_subreddit in class_subreddits:
        for year in years:
                try:
                    year_df = pd.read_csv(f'./{class_subreddit}/{class_subreddit}_all_comments_20{year}.csv', usecols = ['author','author_flair_text'])
                except Exception:
                    pass
                authors = year_df['author'].tolist()
                labels =  year_df['author_flair_text'].tolist()
                authors_list += authors
                labels_list += labels
    author_labels_tup = list(zip(authors_list, labels_list)) #make tuple
    additional_authors_df = pd.DataFrame(author_labels_tup, columns=['authors','labels'])
    print(additional_authors_df.head(10)) # preview the authors data
    additional_authors_df.to_csv(f'./mbti_additional_authors.csv', header=True, index=False, columns=list(additional_authors_df.axes[1]))
    return additional_authors_df
concat_additional_authors()



def clean_add_authors():
    mbti_additional_authors_df = pd.read_csv(f'./mbti_additional_authors.csv')
    mbti_additional_authors_df['labels'] = mbti_additional_authors_df['labels'].str.replace(" ","").str[:4].str.upper()
    mbti_additional_authors_df['labels'] = mbti_additional_authors_df['labels'].str.replace("𝐄𝐒𝐅𝐉","ESFJ").str.replace("𝐈𝐍𝐓𝐏", "INTP")
    
    
    
    #value_counts = mbti_additional_authors_df['labels'].value_counts()
    #to_remove = value_counts[value_counts <= 3].index
    #mbti_additional_authors_df = mbti_additional_authors_df[~mbti_additional_authors_df.labels.isin(to_remove)]
    
    
    
    print(mbti_additional_authors_df.tail(10))
    mbti_additional_authors_df.to_csv(f'./mbti_additional_authors_clean.csv', header=True, index=False, columns=list(mbti_additional_authors_df.axes[1]))
    return mbti_additional_authors_df
clean_add_authors()

def merge_authors():
    mbti_authors_df = pd.read_csv(f'./mbti_authors_cleaned.csv')
    mbti_additional_authors_df = pd.read_csv('./mbti_additional_authors_clean.csv')
    print(f"Total number of main sub authors: {len(mbti_authors_df)}")
    print(f"Total number of sparse sub authors: {len(mbti_additional_authors_df)}")
    enriched_authors_df = pd.concat([mbti_authors_df, mbti_additional_authors_df]) #merge dataframes
    enriched_authors_df = enriched_authors_df.dropna(subset=['labels']) # remove authors without labels (deleted users have no labels)
    print(f"Number of collected authors with label: {len(enriched_authors_df)}")
    enriched_authors_df = enriched_authors_df.drop_duplicates(subset=['authors'])# remove duplicate authors
    enriched_authors_df = enriched_authors_df.groupby('labels').filter(lambda x : len(x)>5)
    print(f"Number of collected authors without duplicates: {len(enriched_authors_df)}")
    enriched_authors_df.to_csv(f'./enriched_authors_cleaned.csv', header=True, index=False, columns=list(enriched_authors_df.axes[1]))
    
    return enriched_authors_df
merge_authors()

import matplotlib.pyplot as plt
authors_df = pd.read_csv(f'./enriched_authors_cleaned.csv')
print(authors_df.tail(50)) 
(
    
    authors_df['labels']
    .value_counts()
    .plot.bar()
)    


plt.style.use('ggplot')
orig_dist = pd.read_csv('./mbti_authors_cleaned.csv')
new_dist = pd.read_csv(f'./enriched_authors_cleaned.csv')
graph_df = orig_dist['labels'].value_counts().rename('Original').to_frame()\
               .join(new_dist['labels'].value_counts().rename('Balanced').to_frame())

graph_df.plot(kind='bar',figsize=(8, 4))
plt.savefig('compare_authors.png')

def add_binary_labels():
    mbti_authors_df = pd.read_csv('mbti_authors_cleaned.csv')
    enriched_authors_df = pd.read_csv(f'./enriched_authors_cleaned.csv')
    
    labels = list(enriched_authors_df['labels'])
    binary1 = [label[0] for label in labels]
    binary2 = [label[1] for label in labels]
    binary3 = [label[2] for label in labels]
    binary4 = [label[3] for label in labels]
    enriched_authors_df = enriched_authors_df.assign(binary1=binary1)
    enriched_authors_df = enriched_authors_df.assign(binary2=binary2)
    enriched_authors_df = enriched_authors_df.assign(binary3=binary3)
    enriched_authors_df = enriched_authors_df.assign(binary4=binary4)
    enriched_authors_df.to_csv('enriched_final_authors.csv', header=True, index=False, columns=list(enriched_authors_df.axes[1]))
    
    labels = list(mbti_authors_df['labels'])
    binary1 = [label[0] for label in labels]
    binary2 = [label[1] for label in labels]
    binary3 = [label[2] for label in labels]
    binary4 = [label[3] for label in labels]
    mbti_authors_df = mbti_authors_df.assign(binary1=binary1)
    mbti_authors_df = mbti_authors_df.assign(binary2=binary2)
    mbti_authors_df = mbti_authors_df.assign(binary3=binary3)
    mbti_authors_df = mbti_authors_df.assign(binary4=binary4)
    print(mbti_authors_df.head(5))
    mbti_authors_df.to_csv('mbti_final_authors.csv', header=True, index=False, columns=list(mbti_authors_df.axes[1]))
    
    return mbti_authors_df, enriched_authors_df
add_binary_labels()




import matplotlib.pyplot as plt
mbti_authors_df = pd.read_csv('./mbti_final_authors.csv')
enriched_authors_df = pd.read_csv('./enriched_final_authors.csv')

(
    
    mbti_authors_df['binary1']
    .value_counts()
    .plot.bar()
)    

(
    
    enriched_authors_df['binary1']
    .value_counts()
    .plot.bar()
)

plt.style.use('ggplot')
orig_dist = pd.read_csv('./mbti_final_authors.csv')
new_dist = pd.read_csv(f'./enriched_final_authors.csv')
"""
(
    orig_dist['binary1'].value_counts().rename('Original').to_frame()
    .join(new_dist['binary1'].value_counts().rename('Balanced').to_frame())
    .join(orig_dist['binary2'].value_counts().rename('Original').to_frame())
    .join(new_dist['binary2'].value_counts().rename('Balanced').to_frame())
    .join(orig_dist['binary3'].value_counts().rename('Original').to_frame())
    .join(new_dist['binary3'].value_counts().rename('Balanced').to_frame())
    .join(orig_dist['binary4'].value_counts().rename('Original').to_frame())
    .join(new_dist['binary4'].value_counts().rename('Balanced').to_frame())
    .plot(kind='bar',figsize=(8, 4))
)
"""

graph_df = orig_dist['binary1'].value_counts().rename('Original').to_frame()\
               .join(new_dist['binary2'].value_counts().rename('Balanced').to_frame())

graph_df.plot(kind='bar',figsize=(8, 4))


graph_df = orig_dist['binary2'].value_counts().rename('Original').to_frame()\
               .join(new_dist['binary2'].value_counts().rename('Balanced').to_frame())

graph_df.plot(kind='bar',figsize=(8, 4))


graph_df.plot(kind='bar',figsize=(8, 4))

graph_df = orig_dist['binary3'].value_counts().rename('Original').to_frame()\
               .join(new_dist['binary3'].value_counts().rename('Balanced').to_frame())

graph_df.plot(kind='bar',figsize=(8, 4))

graph_df = orig_dist['binary4'].value_counts().rename('Original').to_frame()\
               .join(new_dist['binary4'].value_counts().rename('Balanced').to_frame())

graph_df.plot(kind='bar',figsize=(8, 4))
#plt.savefig('compare_authors.png')