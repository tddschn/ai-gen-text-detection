#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://keras.io/img/logo-small.png" alt="Keras logo" width="100"><br/>
# This starter notebook is provided by the Keras team.</center>
# 
# ## Keras NLP starter guide here: https://keras.io/guides/keras_nlp/getting_started/
# 
# In this competition, the challenge is to build a machine learning model that predicts if a text is written by an AI or by a student.
# 
# __This starter notebook uses the [DistilBERT](https://arxiv.org/abs/1910.01108) pretrained model from KerasNLP.__
# 
# 
# **BERT** stands for **Bidirectional Encoder Representations from Transformers**. BERT and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models.
# 
# The BERT family of models uses the **Transformer encoder architecture** to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.
# 
# BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.
# 
# **DistilBERT model** is a distilled form of the **BERT** model. The size of a BERT model was reduced by 40% via knowledge distillation during the pre-training phase while retaining 97% of its language understanding abilities and being 60% faster.
# 
# 
# 
# ![BERT Architecture](https://www.cse.chalmers.se/~richajo/nlp2019/l5/bert_class.png)
# 
# 
# 
# In this notebook, you will:
# 
# - Load the Detect AI Generated Text dataset
# - Explore the dataset
# - Preprocess the data
# - Load a DistilBERT model from Keras NLP
# - Train your own model, fine-tuning BERT
# - Generate the submission file
# 

# In[ ]:


!pip install keras-core --upgrade
!pip install -q keras-nlp
!pip install seaborn

# In[1]:


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras_core as keras
import keras_nlp
import seaborn as sns
import matplotlib.pyplot as plt


print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("KerasNLP version:", keras_nlp.__version__)

# # Load the Detect AI Generated Text
# Let's have a look at all the data files

# In[2]:


DATA_DIR = '/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/'

for dirname, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# # Explore the dataset
# 
# Let's look at the distribution of labels in the training set.

# In[3]:


df_train_prompts = pd.read_csv(DATA_DIR + "train_prompts.csv")
print(df_train_prompts.info())
df_train_prompts.head()

# **Only two prompts are used in this dataset.**
# 
# Let's look at the distribution of text/generated in the training set.

# In[4]:


df_train_essays = pd.read_csv(DATA_DIR + "train_essays.csv")
print(df_train_essays.info())
df_train_essays.head()

# In[5]:


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays,
                   x="prompt_id")

abs_values = df_train_essays['prompt_id'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of prompt ID")

# In[6]:


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays,
                   x="generated")

abs_values = df_train_essays['generated'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of Generated Text")

# **1375 essays are written by human and only 3 by AI.**
# 
# **The distribution between the two prompts is pretty equal.**

# In[7]:


df_test_essays = pd.read_csv(DATA_DIR + "test_essays.csv")
print(df_test_essays.info())
df_test_essays.head()

# In[8]:


df_test_essays["text"].apply(lambda x : len(x))

# **The test dataset contains only 3 essays. The length of each essay is very small (12 characters).**

# # Add new data to the training dataset
# 
# As the dataset does not contain any generated data. We will use the dataset created by [DAREK KŁECZEK](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/455517)

# In[9]:


df_train_essays_ext = pd.read_csv('/mnt/beegfs/xchen87/ai-gen/data/daigt-proper-train-dataset/train_drcat_04.csv')

df_train_essays_ext.rename(columns = {"label":"generated"}, inplace=True)

df_train_essays_ext.info()

# In[10]:


df_train_essays_ext.head()

# In[11]:


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays_ext,
                   x="generated")

abs_values = df_train_essays_ext['generated'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of Generated Text")

# In[12]:


df_train_essays


# In[13]:


df_train_essays_final = pd.concat([df_train_essays_ext[["text", "generated"]], df_train_essays[["text", "generated"]]])

df_train_essays_final.info()

# # Prepare data
# 
# Let's count the number of words in each essay

# In[14]:


df_train_essays["text_length"] = df_train_essays["text"].apply(lambda x : len(x.split()))

# In[15]:


fig = plt.figure(figsize=(40,50))
plot = sns.displot(data=df_train_essays,
                 x="text_length", bins=30, kde=True)
plot.fig.suptitle("Distribution of the length per essay - Train dataset")


# In[16]:


df_train_essays["text_length"].mean() + df_train_essays["text_length"].std()

# # Create the model

# In[17]:


# We choose 512 because it's the limit of DistilBert
SEQ_LENGTH = 512

# Use a shorter sequence length.
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=SEQ_LENGTH,
)

# Pretrained classifier.
classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
    activation=None,
    preprocessor=preprocessor,
)

# Re-compile (e.g., with a new learning rate)
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-4),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy()
   ]
)
    

# Access backbone programmatically (e.g., to change `trainable`).
classifier.backbone.trainable = False


classifier.summary()

# In[ ]:


# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_essays_final["text"],
                                                    df_train_essays_final["generated"],
                                                    test_size=0.33,
                                                    random_state=42)

# In[ ]:


# Fit
classifier.fit(x=X_train, 
               y=y_train,
               validation_data=(X_test, y_test),
               epochs=1,
               batch_size=64
              )

# In[ ]:


def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        np.argmax(y_pred, axis=1),
        display_labels=["Not Generated","Generated"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
    f1_score = tp / (tp+((fn+fp)/2))

    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))


# In[ ]:


y_pred_test = classifier.predict(X_test)

# In[ ]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
displayConfusionMatrix(y_test, y_pred_test,  "Test")

# In[ ]:



