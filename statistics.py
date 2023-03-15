import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
def showstat(dataset, dataset2=None, dataset3=None):
    df1 = dataset.groupby('subreddit').filter(lambda x : len(x)>1000)
    df2 = dataset2.groupby('subreddit').filter(lambda x : len(x)>1000)
    df3 = dataset3.groupby('subreddit').filter(lambda x : len(x)>1000)
    #preprocessed_df = pd.read_feather('preprocessed_df_new.feather')
    #mbti_subreddit_df = pd.read_feather('mbti_subreddit_df.feather')
    #various_subreddit_df = pd.read_feather('various_subreddit_df.feather')
    #display label distribution

    #graph = ( df1['labels'].value_counts().rename('Total').to_frame()
    #    .join(df2['labels'].value_counts().rename('MBTI only').to_frame())
    #    .join(df3['labels'].value_counts().rename('No MBTI').to_frame())
    #    .plot(kind='bar',figsize=(8, 4)))

    #plt.savefig('./plots/compare_subreddit_prop_samples.png')
    #pie_total = df1['subreddit'].value_counts().plot(kind='pie', figsize=(8,4))
    #plt.savefig('./plots/pie_subreddit_prop_total.png')
    #pie_mbti_only = df2['subreddit'].value_counts().plot(kind='pie', figsize=(8,4))
    #plt.savefig('./plots/pie_subreddit_prop_mbtionly.png')
    #pie_nombti = (df3['subreddit']
                    #.value_counts()
                    #.plot(kind='pie', figsize=(8,4))
                    #)
    #plt.savefig('./plots/pie_subreddit_prop_nombti.png')
    print(
        df1['labels']
        .value_counts()
        #.plot.bar(title="Labels")
    )    
    #display subreddit distribution
    print(
        df1['subreddit']
        .value_counts()
        #.plot.bar(title="subreddits")
    )
    # display user distribution
    print(
        df1['authors'].value_counts().rename('Total').to_frame()
    #    .join(df2['authors'].value_counts().rename('MBTI only').to_frame())
    #    .join(df3['authors'].value_counts().rename('No MBTI').to_frame())
    #    .plot(kind='bar',figsize=(8, 4)))
        
        #.plot.bar(title="subreddits")
    )
    print(f"Df 1 No of total authors: {len(set(df1['authors'].tolist()))}")
    print(f"Df 1 No of total authors: {len(set(df2['authors'].tolist()))}")
    print(f"Df 3 No of total authors: {len(set(df3a['authors'].tolist()))}")
    print(f"Df 1 authors mean: {df1['authors'].value_counts().mean()}")
    print(f"Df 1 authors median: {df1['authors'].value_counts().median()}")
    print(f"Df 2 authors mean: {df2['authors'].value_counts().mean()}")
    print(f"Df 2 authors median: {df2['authors'].value_counts().median()}")
    print(f"Df 3 authors mean: {df3['authors'].value_counts().mean()}")
    print(f"Df 3 authors median: {df3['authors'].value_counts().median()}")
    #print(f"Length of dataset: {len(df1)}")

if __name__ == '__main__':
    preprocessed_df = pd.read_feather('preprocessed_df_new.feather')
    mbti_subreddit_df = pd.read_feather('mbti_subreddit_df.feather')
    various_subreddit_df = pd.read_feather('various_subreddit_df.feather')
    disprop_total_100k_df = pd.read_feather('./samples/disprop_sample100k_total.feather')
    prop_total_100k_df = pd.read_feather('./samples/prop_sample_0025_total.feather')
    disprop_mbtionly_50k_df = pd.read_feather('./samples/disprop_sample50k_mbtionly.feather')
    prop_mbtionly_50k_df = pd.read_feather('./samples/prop_sample_016_mbtionly.feather')
    disprop_nombti_df = pd.read_feather('./samples/disprop_sample100k_no_mbti.feather')
    prop_nombti_df = pd.read_feather('./samples/prop_sample_0029_no_mbti.feather')
    
    showstat(prop_total_100k_df, prop_mbtionly_50k_df, prop_nombti_df)
    showstat(disprop_total_100k_df, disprop_mbtionly_50k_df, disprop_nombti_df)