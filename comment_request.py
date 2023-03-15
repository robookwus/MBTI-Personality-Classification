import requests
import time
import pandas as pd
import glob
import csv
import re
import pyarrow.feather as feather
import matplotlib.pyplot as plt
def get_comments_from_pushshift(**kwargs):
    r = requests.get("https://api.pushshift.io/reddit/comment/search/",params=kwargs)
    print(r.url)
    data = r.json()
    return data['data']

def get_comments_from_reddit_api(comment_ids):
    headers = {'User-agent':'Comment Collector'}
    params = {}
    params['id'] = ','.join(["t1_" + id for id in comment_ids])
    r = requests.get("https://api.reddit.com/api/info",params=params,headers=headers)
    data = r.json()
    return data['data']['children']


def collect_comments(start, end):
    """
    collects comments from author from pushshift api in batches of size 100. 
    creates dataframe and saves df in csv.
    """

    authors_df = pd.read_csv(f'./enriched_authors_cleaned.csv')
    



    for i in range(start, end):
        print('\n'+authors_df.at[i, 'authors']+'\n')
        authors_list = []
        labels_list = []
        comments_list = []
        subreddit_name_list = []
        author = authors_df.at[i, 'authors']
        label =  authors_df.at[i, 'labels']

        before = None # checkpoint for batchwise collection
        while True:
            comments = get_comments_from_pushshift(author=author,size=100,before=before,sort_type='created_utc')
            if not comments: break
            #print(comments)
            # This will get the comment ids from Pushshift in batches of 100 -- Reddit's API only allows 100 at a time
            comment_ids = []
            for comment in comments:
                before = comment['created_utc'] # This will keep track of your position for the next call in the while loop
                comment_ids.append(comment['id'])
            
            # This will then pass the ids collected from Pushshift and query Reddit's API for the most up to date information
            comments = get_comments_from_reddit_api(comment_ids)
            for comment in comments:
                comment = comment['data']
                comment_body = comment['body']
                subreddit_name = comment['subreddit_name_prefixed']
                
                print(comment_body)
                
                if comment_body == '[deleted]' or comment_body == ['[removed]']:
                    continue
                else:
                    authors_list.append(author)
                    labels_list.append(label)
                    comments_list.append(comment_body)
                    subreddit_name_list.append(subreddit_name)
            if len(comments_list) > 10000: break # limit comments per user to 10100 max
            time.sleep(3) # I'm not sure how often you can query the Reddit API without oauth but once every two seconds should work fine

        user_comments_tup = list(zip(authors_list, labels_list, comments_list, subreddit_name_list)) #make tuple
        user_comments_df = pd.DataFrame(user_comments_tup, columns=['authors','labels', 'comments', 'subreddit'])
        user_comments_df.to_csv(f'./user_comments/batch{i}_comments_{author}.csv', header=True, index=False, columns=list(user_comments_df.axes[1]))
        print(f"{len(user_comments_df)} comments collected for user {author} with label {label}.")  
        time.sleep(5) # to minimize api overload

def merge_authors():
    """
    merges all author dataframes to one dataframe
    """
    path = "./user_comments/"
    filenames = glob.iglob(path + "/*.csv")
    all_authors_df = pd.DataFrame()

    for fname in sorted(filenames):
        print(fname)
        try:
            user_df = pd.read_csv(fname)
            all_authors_df = pd.concat([all_authors_df, user_df]) # add user to df
        except Exception:
            pass
    print(f"Total Number of collected comments: {len(all_authors_df)}")
    all_authors_df = all_authors_df[all_authors_df.comments != '[deleted]'] # remove deleted comments
    print(f"Total Number of collected comments without deleted comments: {len(all_authors_df)}")
    all_authors_df = all_authors_df[all_authors_df.comments != '[removed]'] # remove removed comments
    print(f"Total Number of collected comments without removed comments: {len(all_authors_df)}")
    all_authors_df = all_authors_df[all_authors_df.comments != 'u/savevideo'] # remove comments with just "u/savevideo"
    print(f"Total Number of collected comments without 'u/savevideo' comments: {len(all_authors_df)}")
    #all_authors_df = all_authors_df[all_authors_df.comments != '[deleted]'] # remove deleted comments
    all_authors_df.to_csv('all_authors_df.csv', header=True, index=False, columns=list(all_authors_df.axes[1]))
    return all_authors_df

def preprocess():
    """
    preprocesses comments by removing short comments and comments starting with a link or 'r/'.
    """
    all_authors_df = pd.read_csv('all_authors_df.csv')
    #all_authors_df = all_authors_df.dropna(subset=['comments']) # remove misaligned comments
    print(f"Total Number of collected comments: {len(all_authors_df)}")
    preprocessed_df = all_authors_df[all_authors_df['comments'].str.len() > 50] #remove comments with less than 50 chars
    print(f"Total Number of collected comments without comments under 50 chars: {len(preprocessed_df)}")
    preprocessed_df = preprocessed_df[preprocessed_df['comments'].str.startswith("http") == False]
    print(f"Total Number of collected comments without comments that start with http: {len(preprocessed_df)}")
    preprocessed_df = preprocessed_df[preprocessed_df['comments'].str.startswith("r/") == False]
    print(f"Total Number of collected comments without comments that start with r/: {len(preprocessed_df)}")
    preprocessed_df.to_csv('preprocessed_df.csv', header=True, index=False, columns=list(preprocessed_df.axes[1]))
    return all_authors_df


def mask_label_tokens():
    """
    Mask labels in text. Also removes html artefacts in text.
    """
    preprocessed_df = pd.read_csv('preprocessed_df.csv')
    preprocessed_df['comments'] = (preprocessed_df['comments']
                                    .str.replace("\&lt;", "")
                                    .str.replace("\&gt;", "")
                                    .str.replace("&amp;", "")
                                    .str.replace("#x200B;", "")
                                    .str.replace("\n", "")
                                    .str.replace(r"(e|E|i|I)(n|N|s|S)(f|F|t|T)(p|P|j|J)", "****")
                                    )

    feather.write_feather(preprocessed_df, 'preprocessed_df_new.feather')




"""
def split_dataset():
    preprocessed_df = pd.read_feather('preprocessed_df_new.feather')
    list_of_mbti_subreddits = ['r/mbti', 'r/intp', 'r/infp', 'r/entp', 'r/intj', 'r/infp', 'r/enfp', 'r/istp', 'r/entj', 'r/estp', 'r/esfp', 'r/enfj', 'r/istj', 'r/esfp', 'r/isfj', 'r/estj','r/esfj', 'r/askISFP', 'r/INTP']
    preprocessed_df = feather.read_feather('preprocessed_df_new.feather')
    mbti_subreddit_df = preprocessed_df[preprocessed_df['subreddit'].str.lower().isin(list_of_mbti_subreddits)]
    print(mbti_subreddit_df.head(20))
    various_subreddit_df = preprocessed_df[~(preprocessed_df['subreddit'].str.strip().isin(list_of_mbti_subreddits) == True)]
    print(various_subreddit_df.head(20))
    feather.write_feather(mbti_subreddit_df,"mbti_subreddit_df.feather")
    feather.write_feather(various_subreddit_df,"various_subreddit_df.feather")
"""


if __name__ == '__main__':
    #start = 2430
    #collect_comments(start, 2431)
    #merge_authors()
    #preprocess()
    #mask_label_tokens()

    #split_dataset()
    

    #sample_dataset(various_subreddit_df)
    """
    # get train sample of first experiment
    #df = pd.read_feather("./preprocessed_df_new.feather")
    #train_sp_df = df.sample(80000, random_state=1)
    #other_df.drop(index=train_sp_df.index)

    disprop_total_100k_df = pd.read_feather('./samples/disprop_sample100k_total.feather')
    prop_total_100k_df = pd.read_feather('./samples/prop_sample_0025_total.feather')
    split_samples(disprop_total_100k_df, frac=0.2, path="./samples/total/")
    split_samples(prop_total_100k_df, frac=0.2, path="./samples/total/")
    
    split_samples(disprop_mbtionly_50k_df, frac=0.2, path="./samples/mbti_only/")
    split_samples(prop_mbtionly_50k_df, frac=0.2, path="./samples/mbti_only/")
    split_samples(disprop_nombti_df, frac=0.2, path="./samples/no_mbti/")
    split_samples(prop_nombti_df, frac=0.2, path="./samples/no_mbti/")
    """
    # group by -> sort -> value at .5
    #remove [deleted], [removed], u/savevideo, under 20chars, 
    # nsfw, multiple languages, comments with "i am <label>",
    # links, alle label maskieren (regex?), remove emojis,
    # &lt; &gt; &amp; #x200B;
    # len(comment.strip()) == 1 and comment.startswith("r/")
    # experiment: train on mbti comments vs train on non-mbti comments