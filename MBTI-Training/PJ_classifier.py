import pandas as pd 

#df = pd.read_feather("//media/data/mbti-reddit/preprocessed_df_new.feather")
df = pd.read_feather("/media/data/mbti-reddit/prop_sample_016_mbtionly.feather") 

df=df.drop(columns=['authors','subreddit'])

df=df.sample(80000, random_state=1) #random sampling


df['labels'] = df['labels'].replace(['ISTP','ESTP','INTP','ENTP','ISFP','ESFP','ENFP','INFP',
                                     'ISTJ','ESTJ','ENTJ','INTJ','ISFJ','ESFJ','ENFJ','INFJ'], \
                                    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]) 
df=df.rename(columns={'labels':'labels','comments':'text'})

from datasets import Dataset

dataset = Dataset.from_pandas(df)
dataset.shuffle(seed=27)
split_set = dataset.train_test_split(test_size=0.2)

from transformers import AlbertTokenizer, AlbertModel

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=16)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = split_set.map(preprocess_function, batched=True)


from transformers import DataCollatorWithPadding
#tokenized_datasets = tokenized_datasets.remove_columns(books_dataset["train"].column_names)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


import evaluate
import numpy as np
def compute_metrics(eval_preds):
    metric = evaluate.combine([
    evaluate.load("precision", average="binary"),
    evaluate.load("recall", average="binary")])
    
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(

    evaluation_strategy="epoch",
    save_strategy="epoch",

    #output_dir="/home/deimann/mbti-project/PJ_train",
    output_dir="/home/deimann/mbti-project/PJ_train_MBTIonly",

    save_total_limit=3,
    load_best_model_at_end = True, 

    learning_rate=2e-5,#2e

    per_device_train_batch_size=9  ,#16

    per_device_eval_batch_size=8,#16

    num_train_epochs=7,

    weight_decay=0.01,

)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["test"],

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

)

trainer.train()