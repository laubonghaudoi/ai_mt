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

# Step 2:
# Start training. Models are saved as `exp-model_acc_X_ppl_X_eX.pt`
python nmt/train.py -data data/exp -save_model exp-model

# Step 3:
# Output inferences.
python translate.py \
-model exp-model_acc_X_ppl_X_eX.pt \
-src data/valid.en \
-output outputs/exp.txt \
-replace_unk \
-verbose