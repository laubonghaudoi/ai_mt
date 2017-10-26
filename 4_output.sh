#! /bin/sh

# Step 3:
# Output inferences.
python nmt/translate.py \
-model ckpts/demo-model_acc_51.94_ppl_12.01_e4.pt \
-src tokenized_jieba_data/test_a.en \
-output outputs/exp.txt \
-batch_size 1024 \
-verbose \
-gpu 0