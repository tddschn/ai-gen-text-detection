# %% [code]
def get_prediction0(model_path, model_weights):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import pickle
    import time
    from torch.cuda.amp import autocast, GradScaler
    import random
    from sklearn.metrics import roc_auc_score, log_loss
    import re

    class DAIGTDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list = text_list
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.text_list)

        def __getitem__(self, index):
            text = self.text_list[index]
            tokenized = self.tokenizer(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            return tokenized["input_ids"].squeeze(), tokenized[
                "attention_mask"
            ].squeeze()

    class DAIGTModel(nn.Module):
        def __init__(self, model_path, config, tokenizer, pretrained=False):
            super().__init__()
            if pretrained:
                self.model = AutoModel.from_pretrained(model_path, config=config)
            else:
                self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            # self.model.gradient_checkpointing_enable()

        def forward_features(self, input_ids, attention_mask=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings

        def forward(self, input_ids, attention_mask):
            embeddings = self.forward_features(input_ids, attention_mask)
            logits = self.classifier(embeddings)
            return logits

    df = pd.read_csv("../input/llm-detect-ai-generated-text/test_essays.csv")
    id_list = df["id"].values
    text_list = df["text"].values

    max_len = 768
    batch_size = 16

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DAIGTModel(model_path, config, tokenizer, pretrained=False)
    model.load_state_dict(torch.load(model_weights))
    model.cuda()
    model.eval()

    test_datagen = DAIGTDataset(text_list, tokenizer, max_len)
    test_generator = DataLoader(
        dataset=test_datagen,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    pred_prob = np.zeros((len(text_list),), dtype=np.float32)
    for j, (batch_input_ids, batch_attention_mask) in tqdm(
        enumerate(test_generator), total=len(test_generator)
    ):
        with torch.no_grad():
            start = j * batch_size
            end = start + batch_size
            if j == len(test_generator) - 1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask)
            pred_prob[start:end] = logits.sigmoid().cpu().data.numpy().squeeze()

    return pred_prob


def get_prediction1():
    import sys
    import gc

    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import numpy as np
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    from tokenizers import (
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
        Tokenizer,
    )

    from datasets import Dataset
    from tqdm.auto import tqdm
    from transformers import PreTrainedTokenizerFast

    from sklearn.linear_model import SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import VotingClassifier

    test = pd.read_csv(
        "/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/test_essays.csv"
    )
    sub = pd.read_csv(
        "/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/sample_submission.csv"
    )
    org_train = pd.read_csv(
        "/mnt/beegfs/xchen87/ai-gen/data/llm-detect-ai-generated-text/train_essays.csv"
    )
    train = pd.read_csv(
        "/mnt/beegfs/xchen87/ai-gen/data/daigt-v2-train-dataset/train_v2_drcat_02.csv",
        sep=",",
    )

    train = train.drop_duplicates(subset=["text"])
    train.reset_index(drop=True, inplace=True)

    LOWERCASE = False
    VOCAB_SIZE = 30522

    # Creating Byte-Pair Encoding tokenizer
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else []
    )
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
    dataset = Dataset.from_pandas(test[["text"]])

    def train_corp_iter():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenized_texts_test = []

    for text in tqdm(test["text"].tolist()):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    tokenized_texts_train = []

    for text in tqdm(train["text"].tolist()):
        tokenized_texts_train.append(tokenizer.tokenize(text))

    def dummy(text):
        return text

    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        analyzer="word",
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents="unicode",
    )

    vectorizer.fit(tokenized_texts_test)

    # Getting vocab
    vocab = vectorizer.vocabulary_

    print(vocab)

    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        vocabulary=vocab,
        analyzer="word",
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents="unicode",
    )

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer
    gc.collect()

    y_train = train["label"].values

    if len(test.text.values) <= 0:
        sub.to_csv("submission.csv", index=False)
    else:
        clf = MultinomialNB(alpha=0.02)
        sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
        p6 = {
            "n_iter": 2500,
            "verbose": -1,
            "objective": "cross_entropy",
            "metric": "auc",
            "learning_rate": 0.00581909898961407,
            "colsample_bytree": 0.78,
            "colsample_bynode": 0.8,
            "lambda_l1": 4.562963348932286,
            "lambda_l2": 2.97485,
            "min_data_in_leaf": 115,
            "max_depth": 23,
            "max_bin": 898,
        }

        lgb = LGBMClassifier(**p6)
        cat = CatBoostClassifier(
            iterations=2000,
            verbose=0,
            l2_leaf_reg=6.6591278779517808,
            learning_rate=0.005599066836106983,
            subsample=0.4,
            allow_const_label=True,
            loss_function="CrossEntropy",
        )

        weights = [0.068, 0.31, 0.31, 0.312]

        ensemble = VotingClassifier(
            estimators=[("mnb", clf), ("sgd", sgd_model), ("lgb", lgb), ("cat", cat)],
            weights=weights,
            voting="soft",
            n_jobs=-1,
        )
        ensemble.fit(tf_train, y_train)
        gc.collect()
        final_preds = ensemble.predict_proba(tf_test)[:, 1]

    return final_preds


def main():
    import numpy as np
    import pandas as pd
    import os

    model_path = "../input/daigtconfigs/debertav3large"
    model_weights = "../input/daigttrained11/17_ft1/weights_ep0"
    pred_prob0 = get_prediction0(model_path, model_weights)

    model_path = "../input/daigtconfigs/debertav3large"
    model_weights = "../input/daigttrained11/17_ft103/weights_ep0"
    pred_prob0 += get_prediction0(model_path, model_weights)

    model_path = "../input/daigtconfigs/debertav3large"
    model_weights = "../input/daigttrained11/19_ft1/weights_ep0"
    pred_prob0 += get_prediction0(model_path, model_weights)

    model_path = "../input/daigtconfigs/debertav3large"
    model_weights = "../input/daigttrained11/19_ft103/weights_ep0"
    pred_prob0 += get_prediction0(model_path, model_weights)

    model_path = "../input/daigtconfigs/debertalarge"
    model_weights = "../input/daigttrained11/20_ft1/weights_ep0"
    pred_prob0 += get_prediction0(model_path, model_weights)

    model_path = "../input/daigtconfigs/debertalarge"
    model_weights = "../input/daigttrained11/20_ft103/weights_ep0"
    pred_prob0 += get_prediction0(model_path, model_weights)

    pred_prob0 /= 6.0

    pred_prob1 = get_prediction1()

    pred_prob = (
        0.625 * pred_prob0.argsort().argsort() + 0.375 * pred_prob1.argsort().argsort()
    )

    df = pd.read_csv("../input/llm-detect-ai-generated-text/test_essays.csv")
    id_list = df["id"].values
    sub_df = pd.DataFrame(data={"id": id_list, "generated": pred_prob})
    sub_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
