__author__ = "Shaptarshi Roy"

def sample_dataset(dataset):
    df= dataset
    print(
        df['labels']
        .value_counts()
        #.plot.bar(title="Labels")
    )
    print(len(df))
    
    sample_size=6250
    #disproportionate sample, 100k total
    disprop_sample100k = df.groupby('labels').apply(lambda x: x.sample(sample_size, random_state=1))
    feather.write_feather(disprop_sample100k,"./samples/disprop_sample100k_no_mbti.feather") # change depending on sample
    print(disprop_sample100k.head(20))
    #proportionate sample, around 100k total
    prop_sample_0029 =  df.groupby('labels').apply(lambda x: x.sample(frac=0.029, random_state=1))
    print(prop_sample_0029.head(20))
    feather.write_feather(prop_sample_0029,"./samples/prop_sample_0029_no_mbti.feather") #change depending on sample
    
def split_samples(df, frac, path):
    test = df.sample(frac=frac, axis=0, random_state=1)
    train = df.drop(index=test.index)
    feather.write_feather(test, f"{path}{df}_test.feather")
    feather.write_feather(train, f"{path}{df}_train.feather")

if __name__ == '__main__':
    total_subreddit_df = pd.read_feather('preprocessed_df_new.feather')
    mbti_subreddit_df = pd.read_feather('mbti_subreddit_df.feather')
    various_subreddit_df = pd.read_feather('various_subreddit_df.feather')
    sample_dataset(total_subreddit_df)
    sample_dataset(mbti_subreddit_df)
    sample_dataset(various_subreddit_df)