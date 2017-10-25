#! /bin/sh


# Step 1:
# Read source and target data and save as OpenNMT torchtext objects
# Train file is save as `data/exp.train.pt`, validation file `data/exp.valid.pt``
python nmt/preprocess.py \
-train_src tokenized_jieba_data/train.en \
-train_tgt tokenized_jieba_data/train.zh \
-valid_src tokenized_jieba_data/valid.en \
-valid_tgt tokenized_jieba_data/valid.zh \
-save_data data/exp
