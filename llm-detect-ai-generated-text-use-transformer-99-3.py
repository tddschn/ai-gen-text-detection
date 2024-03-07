#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#install libs
import pickle
from leven_search import LevenSearch, EditCost, EditCostConfig, GranularEditCostConfig

with open('/kaggle/usr/lib/install-levenshtein-search-library/leven_search.pkl', 'rb') as file:
    lev_search = pickle.load(file)


# In[ ]:


#Add all imports 
import os
import gc
import re
import sys
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import ( decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer,)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter
from itertools import chain

from tqdm.auto import tqdm

from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression  #0.925
# from sklearn.linear_model import SGDClassifier #0.42
from sklearn.preprocessing import MaxAbsScaler
# from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression


from sklearn.svm import LinearSVR
from scipy.sparse import vstack as spvstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from transformers import EarlyStoppingCallback
from collections import defaultdict

from tqdm import tqdm
tqdm.pandas()

#disable wandb
import wandb
wandb.init(mode="disabled")

# In[ ]:


#spell checker code 
def sentence_correcter(text):
    wrong_words = []
    correct_words = dict()

    word_list = re.findall(r'\b\w+\b|[.,\s]', text)

    for t in word_list:
        correct_word = t

        if len(t)>2:
            result = lev_search.find_dist(t, max_distance=0)
            result = list(result.__dict__['words'].values())

            if len(result) == 0:
                result = lev_search.find_dist(t, max_distance=1)
                result = list(result.__dict__['words'].values())
                if len(result):
                    correct_word = result[0].word
                    wrong_words.append((t, result))

        correct_words[t] = correct_word

    dict_freq = defaultdict(lambda :0)           
    for wrong_word in wrong_words:
        _, result = wrong_word

        for res in result:
            updates = res.updates
            parts = str(updates[0]).split(" -> ")
            if len(parts) == 2:
                from_char = parts[0]
                to_char = parts[1]
                dict_freq[(from_char, to_char)] += 1

    if len(dict_freq):
        max_key = max(dict_freq, key=dict_freq.get)
        count = dict_freq[max_key]
    else:
        count = 0

    if count > 0.06*len(text.split()):
        gec = GranularEditCostConfig(default_cost=10, edit_costs=[EditCost(max_key[0], max_key[1], 1)])

        for wrong_word in wrong_words:
            word, _ = wrong_word
            result = lev_search.find_dist(word, max_distance=9, edit_cost_config=gec)
            result = list(result.__dict__['words'].values())
            if len(result):
                correct_words[word] = result[0].word
            else:
                correct_word = word


    correct_sentence = []
    for t in word_list:
        correct_sentence.append(correct_words[t])

    return "".join(correct_sentence)

# In[ ]:


#define general use methods 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  


model_checkpoint_base = "/mnt/beegfs/xchen87/ai-gen/data/distilroberta-base/distilroberta-base"
model_checkpoint_infer = "/mnt/beegfs/xchen87/ai-gen/data/detect-llm-models/distilroberta-finetuned_v5/checkpoint-13542"

tokenizer_infer = AutoTokenizer.from_pretrained(model_checkpoint_infer)
tokenizer_train = AutoTokenizer.from_pretrained(model_checkpoint_base)


# In[ ]:


# define data reading methods 

def read_sub():
    return pd.read_csv('/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/sample_submission.csv')

def read_test():
        return pd.read_csv('/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/test_essays.csv')

def read_dummy_test():
    t =  pd.read_csv('/mnt/beegfs/xchen87/ai-gen/data/daigt-v2-train-dataset/train_v2_drcat_02.csv').tail(9000)
    t['id'] = range(0, len(t))
    t = t[['id', 'text']]
    return t


def append_train_from_sub_phase(org_train_data, train_from_sub):
    
    train_from_sub.drop('generated', axis=1, inplace=True)
    train_from_sub.reset_index(drop=True, inplace=True)

    train_from_sub = train_from_sub[['text', 'label']]
    
    train =  pd.concat([org_train_data, train_from_sub])
    
    return train


# In[ ]:


#define readig for text categorization
def filter_dataframe(df, category):
    # Filter the DataFrame for the specified category or NaN in 'prompt_name'
    filtered_df = df[(df['prompt_name'] == category) | (df['prompt_name'].isna())]
    return filtered_df

def filter_dataframe_single_category(df, category):
    # Filter the DataFrame for the specified category in 'prompt_name'
    filtered_df = df[df['prompt_name'] == category]
    return filtered_df

def standardize_categories(df):
    # Standardize the category name
    df['prompt_name'] = df['prompt_name'].str.replace('"A Cowboy Who Rode the Waves"', 'A Cowboy Who Rode the Waves', regex=False)
    return df

def assign_category(row):
    if row['prompt_id'] == 1:
        return "Does the electoral college work?"
    elif row['prompt_id'] == 0:
        return "Car-free cities"
    else:
        return None  # or some default value

    
def read_train_all():
    train = pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/daigt-v2-train-dataset/train_v2_drcat_02.csv")
    train = train[['text', 'prompt_name', 'label']]
    train = standardize_categories(train)

    train_old =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/train_essays.csv")
    train_old.rename(columns={'generated': 'label'}, inplace=True)
    train_old['prompt_name'] = train_old.apply(assign_category, axis=1)
    train_old = train_old[['text', 'prompt_name', 'label']]
    
    lm_7b =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-mistral-7b-instruct-texts/Mistral7B_CME_v7.csv")
    lm_ali_1 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_fac_v1.csv")
    #lm_ali_2 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_elec_v1.csv")
    #lm_ali_3 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_car_free_v1.csv")
    lm_ali_4 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_exploring_venus_v1.csv")
    lm_ali_5 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_face_on_mars_v1.csv")
    lm_ali_6 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_driveless_cars_v1.csv")
    lm_ali_7 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_cowboy_v1.csv")
    lm_ali_8 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_cowboy_v2.csv")
    #lm_ali_9 =  pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/llm-dataset/gen_llm_face_on_mars_v2.csv")
    gemini = pd.read_csv("/mnt/beegfs/xchen87/ai-gen/data/gemini-pro-llm-daigt/gemini_pro_llm_text.csv")
    gemini = gemini[gemini['typos']=="no"]
    #lm_data = pd.concat([lm_7b, lm_ali_1, lm_ali_2, lm_ali_3,lm_ali_4,lm_ali_5,lm_ali_6,lm_ali_7,lm_ali_8,lm_ali_9,gemini], ignore_index=True)
    lm_data = pd.concat([lm_7b, lm_ali_1, lm_ali_4,lm_ali_5,lm_ali_6,lm_ali_7,lm_ali_8,lm_ali_8,gemini], ignore_index=True)

    lm_data.rename(columns={'generated': 'label'}, inplace=True)
    del gemini
    gc.collect()
    lm_data = lm_data[['text', 'prompt_name', 'label']]
    lm_data = standardize_categories(lm_data)
    train_old = standardize_categories(train_old)
    
    train =  pd.concat([train, lm_data, train_old])
    return train, lm_data, train_old

# In[ ]:


# define PBE Tokenizer class and related methods

class BPETokenizer:
    ST = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    def __init__(
        self,
        vocab_size,
    ):
        self.vocab_size = vocab_size
        self.tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.normalizer = normalizers.Sequence([normalizers.NFC()])
        self.tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        
    @classmethod
    def chunk_dataset(cls, dataset, chunk_size=1_000):
        for i in range(0, len(dataset), chunk_size):
            yield dataset[i : i + chunk_size]["text"]
        
    def train(self, data):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.ST)
        dataset = Dataset.from_pandas(data[["text"]])
        self.tok.train_from_iterator(self.chunk_dataset(dataset), trainer=trainer)
        return self
    
    def tokenize(self, data):
        tokenized_texts = []
        for text in tqdm(data['text'].tolist()):
            tokenized_texts.append(self.tok.encode(text))
        return tokenized_texts
    
def train_tokenizer(train, lm_data, train_old, test):

    tokenizer_train_data = pd.concat([lm_data,train_old])
    tok_data = pd.concat([ tokenizer_train_data[["text"]],  test[["text"]] ]).reset_index(drop=True)
    vc_counters = {}
    for vs in [5_000]: # for if needed to train multi vocab_size 
        bpe_tok = BPETokenizer(vs).train(tok_data)
        ctr = Counter(chain(*[x.ids for x in bpe_tok.tokenize(tok_data)]))
        vc_counters[vs] = (bpe_tok, ctr)
        tqdm.write(f"completed tokenization with {vs:,} vocab size")
    return vc_counters

def tokenize_datasets (vc_counters, train, lm_data, test):
    
    bpe_tok = vc_counters[5_000][0]
    test_extend = pd.concat([lm_data,test])
    tokenized_texts_train = [x.tokens for x in bpe_tok.tokenize(train)]
    tokenized_texts_test = [x.tokens for x in bpe_tok.tokenize(test)]
    tokenized_texts_lm_data = [x.tokens for x in bpe_tok.tokenize(lm_data)]
    tokenized_texts_test2 = tokenized_texts_lm_data + tokenized_texts_test
    
    del tokenized_texts_lm_data
    gc.collect()
    
    return tokenized_texts_train, tokenized_texts_test, tokenized_texts_test2


# In[ ]:


#def vectorizer related methods 
def dummy1(text):
    return text

def vectorizer_of_data(tokenized_texts_train,tokenized_texts_test,tokenized_texts_test2, min_diff, test):
    len_test = len(test)
    #print(len_test)
    if len_test < 10:
        min_diff = 0
    
    print("vectorizer - prepare vocab ..")
    vectorizer = TfidfVectorizer(ngram_range=(3, 6), lowercase=False, sublinear_tf=True, analyzer = 'word',min_df=min_diff, 
                                 tokenizer = dummy1, preprocessor = dummy1, token_pattern = None, strip_accents='unicode')
    
    vectorizer.fit(tokenized_texts_test2)
    vocab = vectorizer.vocabulary_
    
    del vectorizer
    gc.collect()
    
    vectorizer = TfidfVectorizer(ngram_range=(3, 6), lowercase=False, sublinear_tf=True, vocabulary=vocab, analyzer = 'word',
                                 tokenizer = dummy1, preprocessor = dummy1, min_df=min_diff, token_pattern = None,
                                 strip_accents='unicode')

    print("vectorizer - fit transform on train ..")

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    
    print("vectorizer - transform on test ..")

    tf_test = vectorizer.transform(tokenized_texts_test)
    
    del tokenized_texts_test2
    del tokenized_texts_test
    del tokenized_texts_train
    
    del vectorizer
    gc.collect()

    return tf_train, tf_test

# In[ ]:


#methods to adopt learning from test data

def build_new_train_from_sub_all(final_preds_linear_tmp, test, X, Y):
    
    test.loc[:, 'generated'] = final_preds_linear_tmp
    sorted_df = test.sort_values(by='generated', ascending=False)
    top_rows = sorted_df.head(X).copy()
    top_rows['label'] = 1
    
    # Select the bottom Y rows and set 'generated' to 0
    bottom_rows = sorted_df.tail(Y).copy()
    bottom_rows['label'] = 0
    
    # Concatenate the two subsets
    train_from_sub = pd.concat([top_rows, bottom_rows])
    
    return train_from_sub

def build_new_train_from_sub_by_classs(final_preds_linear_tmp, test, X, Y):
    test.loc[:, 'generated'] = final_preds_linear_tmp
    class_dfs = {}
    
    # Iterate over each unique class value and create a separate DataFrame
    for class_value in test['prompt_id'].unique():
        class_dfs[class_value] = test[test['prompt_id'] == class_value]
    #print(class_dfs[class_value])
    
    sorted_class_dfs = {class_value: df.sort_values(by='generated', ascending=False) for class_value, df in class_dfs.items()}
    
    new_class_dfs_with_generated = {}
    
    new_class_dfs_filtered = {}
    for class_value, df in sorted_class_dfs.items():
        if len(df) >= (X + Y):
            top_rows = df.head(X).copy()
            top_rows['label'] = 1
            
            bottom_rows = df.tail(Y).copy()
            bottom_rows['label'] = 0
            
            combined = pd.concat([top_rows, bottom_rows], axis=0)
            new_class_dfs_filtered[class_value] = combined
            
    if len(test) > 10:
        train_from_sub = pd.concat(new_class_dfs_filtered.values(), ignore_index=True)
    else:
        train_from_sub = test
        test['label'] = 1

    return train_from_sub


def build_new_train_from_sub_add_all_data(final_preds_linear_tmp, test):
    
    test.loc[:, 'generated'] = final_preds_linear_tmp
    median_label = test['generated'].median()
    test['label'] = (test['generated'] >= median_label).astype(int)
    train_from_sub = test.copy()
    return train_from_sub


# In[ ]:


# define linear models training and prediction methods
def MaxAbsScalerTransform(tf_train,tf_test):
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(tf_train)
    X_test_scaled = scaler.transform(tf_test)
    return X_train_scaled, X_test_scaled
    

def get_predictions_linear_LinearSVR(X_train_scaled,X_test_scaled, y_train):
    
    #X_train_scaled, X_test_scaled = MaxAbsScalerTransform(tf_train,tf_test, y_train)
    
    model = Ridge(solver='sag',max_iter=10000,tol=1e-4, alpha = 1)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled.copy())
    preds = sigmoid(preds)
    
    return preds

def get_predictions_linear_Ridge(X_train_scaled,X_test_scaled, y_train):
    
    #X_train_scaled, X_test_scaled = MaxAbsScalerTransform(tf_train,tf_test, y_train)
    
    model = Ridge(solver='sag',max_iter=8000,tol=1e-4, alpha = 1)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled.copy())
    preds = sigmoid(preds)
    
    return preds

# In[ ]:


#define needed function and variables for transformer model used.

def preprocess_function_infer(examples):
    return tokenizer_infer(examples['text'], max_length = 512 , padding=True, truncation=True)

def preprocess_function_train(examples):
    return tokenizer_train(examples['text'], max_length=128, padding=True, truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    auc = roc_auc_score(labels, probs[:,1], multi_class='ovr')
    return {"roc_auc": auc}



# In[ ]:


#define transformer model inference method

def get_predictions_tranformer(test):
    #model_checkpoint_infer = "/mnt/beegfs/xchen87/ai-gen/data/detect-llm-models/distilroberta-finetuned_v5/checkpoint-13542"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_infer, num_labels=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_ds = Dataset.from_pandas(test)
    test_ds_enc = test_ds.map(preprocess_function_infer, batched=True)
    trainer = Trainer(model,tokenizer=tokenizer_infer,)
    test_preds = trainer.predict(test_ds_enc)
    logits = test_preds.predictions
    final_preds_trans = sigmoid(logits)[:,0]
    
    return final_preds_trans


# In[ ]:


#define transformer model training method
def train_inference_transformer_runtime(train,train_from_sub,test):
    
    #train = append_train_from_sub_phase(train, train_from_sub)
    valid = train_from_sub
    test = test[['id', 'text']]

#     sk = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
#     train0 = train
#     for i, (tr,val) in enumerate(sk.split(train0,train0.label)):
#         train = train0.iloc[tr]
#         valid = train0.iloc[val]
#         break
        
    train.text = train.text.fillna("")
    valid.text = valid.text.apply(lambda x: x.strip('\n'))
    train.text = train.text.apply(lambda x: x.strip('\n'))
    
    ds_train = Dataset.from_pandas(train)
    ds_valid = Dataset.from_pandas(valid)
    
    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_base)
    ds_train_enc = ds_train.map(preprocess_function_train, batched=True)
    ds_valid_enc = ds_valid.map(preprocess_function_train, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_base, num_labels=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
    
    num_train_epochs=10.0
    metric_name = "roc_auc"
    model_name = "distilroberta"
    batch_size = 2
    
    args = TrainingArguments(
        f"{model_name}-finetuned_v5",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        lr_scheduler_type = "cosine",
        save_safetensors = False,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=ds_train_enc,
        eval_dataset=ds_valid_enc,
        tokenizer=tokenizer_train,
        callbacks = [early_stopping],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    trained_model = trainer.model
    
    test_ds = Dataset.from_pandas(test)
    test_ds_enc = test_ds.map(preprocess_function_infer, batched=True)
    trainer = Trainer(trained_model,tokenizer=tokenizer_train,)
    test_preds = trainer.predict(test_ds_enc)
    logits = test_preds.predictions
    probs = sigmoid(logits)[:,1]
    
    return probs

# In[ ]:


#methods for selecting weak rows from test data 

def get_selected_rows_from_test(test, X, final_predictions, final_preds_trans):
    test = read_test()
    test['label'] = final_predictions
    test['trans_label'] = final_preds_trans
    test = test[['id','text', 'label','trans_label']].copy()


    # Calculate the median of the label values
    median_label = test['label'].median()

    # Compute the absolute difference from the median for each row
    test['difference_from_median'] = abs(test['label'] - median_label)

    # Sort the DataFrame by this difference
    test_sorted = test.sort_values('difference_from_median')

    # Select the top X rows (replace X with your desired number of rows)
    #X = 50  # For example, to select 10 rows
    selected_rows = test_sorted.head(X)
    selected_rows = selected_rows.drop(columns=['difference_from_median'])

    return selected_rows

def get_similar_train_text(selected_rows, number_of_similar_rows, train, lm_data, train_old):
    #train, lm_data, train_old = read_train(only_7_prompts = True)

    print("train tokenizer ... iter: similarity")
    vs_counters = train_tokenizer(train, lm_data, train_old, selected_rows)

    print("tokenize datasets ... iter: similarity")
    tokenized_texts_train, tokenized_texts_test, tokenized_texts_test2 = tokenize_datasets (vs_counters, train, lm_data, selected_rows)

    print("verctorize datasets ... similarity ")
    tfidf_train, tfidf_selected = vectorizer_of_data(tokenized_texts_train,tokenized_texts_test,tokenized_texts_test2, 2, test )

    del tokenized_texts_test2, tokenized_texts_test, tokenized_texts_train, vs_counters
    gc.collect()

    similar_texts_dfs = []

    # Iterate over the TF-IDF vectors of selected_rows
    for i in range(tfidf_selected.shape[0]):
        # Calculate cosine similarity between the selected text and all train texts
        cosine_similarities = cosine_similarity(tfidf_selected[i], tfidf_train)

        # Sort the similarities and get the top ones (e.g., top 10)
        top_indices = cosine_similarities.argsort()[0][-number_of_similar_rows:]  # Adjust number as needed

        # Retrieve the similar texts from train_df
        similar_texts = train.iloc[top_indices]

        # Add the DataFrame of similar texts to the list
        similar_texts_dfs.append(similar_texts)

    # Concatenate all the DataFrames in the list to form the final DataFrame
    similar_texts_df = pd.concat(similar_texts_dfs).drop_duplicates()

    return similar_texts_df

# In[ ]:


#define training X times strong procedures

def Train_Linear_X_Times_With_Infer_STRONG(number_of_times, X_TOP, Y_BOTTOM, test_data, only_7_prompts, trans_predictions, test_in,train_in, lm_data_in, train_old_in):
    
    final_preds_trans_DistilRoberta = trans_predictions
    first_time = True
    feedback_times = number_of_times
    final_predictions = []
    X = X_TOP
    Y = Y_BOTTOM
    
    train_from_sub = read_test() #dummy def
    
    for i in range(1, feedback_times+1):
        print("reading datasets ... iter: ", i)
        #sub = read_sub()
        #test = test_data  #read_dummy_test() #read_test()
        #train, lm_data, train_old = read_train(only_7_prompts = only_7_prompts)
        test = test_in.copy()
        train = train_in.copy()
        lm_data = lm_data_in.copy()
        train_old = train_old_in.copy()
    
        if first_time == False:
            train = append_train_from_sub_phase(train, train_from_sub)
        
        print("train tokenizer ... iter: ", i)
        vs_counters = train_tokenizer(train, lm_data, train_old, test)
    
        print("tokenize datasets ... iter: ", i)
        tokenized_texts_train, tokenized_texts_test, tokenized_texts_test2 = tokenize_datasets (vs_counters, train, lm_data, test)
    
        print("verctorize datasets ...iter: ", i)
        tf_train, tf_test = vectorizer_of_data(tokenized_texts_train,tokenized_texts_test,tokenized_texts_test2, 2, test )
        
        del tokenized_texts_test2, tokenized_texts_test, tokenized_texts_train, vs_counters
        gc.collect()
    
        print("predictions ... iter: ", i)
        X_train_scaled, X_test_scaled = MaxAbsScalerTransform(tf_train,tf_test)
        final_preds_linear_tmp = get_predictions_linear_LinearSVR(X_train_scaled, X_test_scaled, train['label'].values)
    
        del tf_train, tf_test, X_train_scaled, X_test_scaled 
        gc.collect()
    
        print("predeictions from iter : ",i, " is : ", final_preds_linear_tmp)
        
        if first_time == True:
            final_preds_phase_tmp = 0.5*final_preds_linear_tmp + 0.5*final_preds_trans_DistilRoberta
        else:
            final_preds_phase_tmp = 0.65*final_preds_linear_tmp + 0.35*final_preds_trans_DistilRoberta
    
        #final_preds_phase_tmp = 0.5*final_preds_linear_tmp + 0.5*final_preds_trans_DistilRoberta
    
        print("final predeictions from iter : ",i, " is : ", final_preds_phase_tmp)

        final_predictions = final_preds_phase_tmp #final_preds_phase_tmp
    
        train_from_sub = build_new_train_from_sub_all(final_preds_phase_tmp, test, X, Y)
        X = X + int(250/i)
        Y = Y + int(250/i)
        print("new x,y: ",X,Y)
    
        first_time = False
        
    return final_predictions, train_from_sub
    

# In[ ]:


#define training X times weak procedures
def Train_Linear_With_Infer_WEAK(X_MIDDLE,SIMILAR_ROWS, test_data, only_7_prompts, final_predictions, final_preds_trans, train, lm_data, train_old):
    
    test = test_data
    selected_rows = get_selected_rows_from_test(test,X_MIDDLE,final_predictions, final_preds_trans)
    similar_texts_df = get_similar_train_text(selected_rows, SIMILAR_ROWS,train, lm_data, train_old)
    
    #train, lm_data, train_old = read_train(only_7_prompts = only_7_prompts)
    train = similar_texts_df
    train = train.drop_duplicates(subset='text', keep='first')
    train = train[['text', 'label']]
    train.reset_index(drop=True, inplace=True)
        
    test = selected_rows 
        
    print("train tokenizer ... iter: over weak")
    vs_counters = train_tokenizer(train, lm_data, train_old, test)
        
    print("tokenize datasets ... iter: over weak")
    tokenized_texts_train, tokenized_texts_test, tokenized_texts_test2 = tokenize_datasets (vs_counters, train, lm_data, test)
        
    print("verctorize datasets ...iter: over weak")
    tf_train, tf_test = vectorizer_of_data(tokenized_texts_train,tokenized_texts_test,tokenized_texts_test2, 2 ,test)
        
    del tokenized_texts_test2, tokenized_texts_test, tokenized_texts_train, vs_counters
    gc.collect()
        
    print("predictions ... iter: over weak")
    X_train_scaled, X_test_scaled = MaxAbsScalerTransform(tf_train,tf_test)
    preds_linear_weak= get_predictions_linear_LinearSVR(X_train_scaled, X_test_scaled, train['label'].values)
    
    del tf_train, tf_test, X_train_scaled, X_test_scaled 
    gc.collect()
    
    trans_sel_pred = selected_rows['trans_label']
    trans_sel_pred = trans_sel_pred.values
    selected_rows_final_pred = 0.5*preds_linear_weak +0.5*trans_sel_pred
    selected_rows['label'] = selected_rows_final_pred
    
    test = read_test()
    test['label'] = final_predictions
    
    test.set_index('id', inplace=True, drop=True)
    selected_rows.set_index('id', inplace=True, drop=True)
    # Update test_df with the new label values from selected_rows
    test.update(selected_rows)
    # Reset index if you want to revert 'id' back to a column
    test.reset_index(inplace=True)

    return test['label']


# In[ ]:


def train_for_prompt_names():
    train_for_cat, lm_data, train_old = read_train_all()
    tfidf_cat = TfidfVectorizer()
    X_cat = tfidf_cat.fit_transform(train_for_cat['text'])
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, train_for_cat['prompt_name'], test_size=0.05, random_state=42)
    model_cat = LogisticRegression(max_iter=1000)
    model_cat.fit(X_train_cat, y_train_cat)
    return tfidf_cat, model_cat


def get_test_prompt_names(test,tfidf_cat,model_cat):
    unique_prompt_ids_count = test['prompt_id'].nunique()
    print("There are : ", unique_prompt_ids_count, " prompts in the test data")

    predefined_prompt_names_list = ['Facial action coding system', 'Driverless cars', 'Exploring Venus', 'The Face on Mars', 'A Cowboy Who Rode the Waves']
    if len(test) == 3:
        print("select predefined prompts .. ")
        return predefined_prompt_names_list

    # predict test cats category 
    test_cats = tfidf_cat.transform(test['text'])
    test_prompt_names = model_cat.predict(test_cats)
    test['prompt_name'] = test_prompt_names
    # get top n
    prompt_counts = test['prompt_name'].value_counts()
    top_n_prompts = prompt_counts.head(unique_prompt_ids_count).index.tolist()
    return top_n_prompts


# In[ ]:


#check if run is for save or submit
submit_mode = True
if len(pd.read_csv('/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/test_essays.csv')) == 3:
    submit_mode = False

# In[ ]:


submit_mode = True


# In[ ]:


if submit_mode == True:
    print("Predict the prompt names used in test ...")
    test = read_test()
    tfidf_cat, model_cat = train_for_prompt_names()
    top_n_prompts = get_test_prompt_names(test,tfidf_cat,model_cat)
    print(top_n_prompts)
    
    print("done ...")
    print("select training data based on the prompt names .. ")
    
    train_all, lm_data, train_old_all = read_train_all()
    train = train_all[train_all['prompt_name'].isin(top_n_prompts)]
    train = pd.concat([train, train_old_all] , ignore_index=True)
    #lm_data = lm_data_all[lm_data_all['prompt_name'].isin(top_n_prompts)]
    train_old = train_old_all.copy()
    
    test = read_test()
    #train, lm_data, train_old = read_train(only_7_prompts = True)
    
    print("spelling process...")
    train['text'] = train['text'].progress_apply(sentence_correcter)
    lm_data['text'] = lm_data['text'].progress_apply(sentence_correcter)
    train_old['text'] = train_old['text'].progress_apply(sentence_correcter)
    test['text'] = test['text'].progress_apply(sentence_correcter)

    print("predictions from transformer")
    final_preds_trans_DistilRoberta = get_predictions_tranformer(read_test()) # you may want to try with corrected test data 
    print ("Predictions from Trans: ", final_preds_trans_DistilRoberta)
    final_predictions_strong, train_from_sub = Train_Linear_X_Times_With_Infer_STRONG(4,1000,1500,test,True, final_preds_trans_DistilRoberta, test,train, lm_data, train_old)
    
    print ("Predictions from Strong: ", final_predictions_strong)
    final_predictions_strong_weak = Train_Linear_With_Infer_WEAK(50,100, test, True, final_predictions_strong, final_preds_trans_DistilRoberta, train, lm_data, train_old)
    print ("Predictions from Weak: ", final_predictions_strong_weak)
    
    #train, lm_data, train_old = read_train(only_7_prompts = True)  #this might affect the output if was droped
    final_preds_trans_DistilRoberta_with_test_feedback = train_inference_transformer_runtime(train,train_from_sub,test)
    print ("Predictions from trans with feedback: ", final_preds_trans_DistilRoberta_with_test_feedback)
    
    final_predictions_deep = 0.8*final_predictions_strong_weak+0.2*final_preds_trans_DistilRoberta_with_test_feedback
    
    sub = read_sub() #read_dummy_test() #read_sub()

    sub['generated'] = final_predictions_deep
    sub.to_csv('submission.csv', index=False)
    
else:
    sub = read_sub()
    sub['generated'] = 1
    sub.to_csv('submission.csv', index=False)
    
    
    
    
