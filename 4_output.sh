#! /bin/sh

# Step 3:
# Output inferences.
python nmt/translate.py \
-model ckpts/demo-model_acc_54.09_ppl_9.76_e13.pt \
-src tokenized_jieba_data/test_a.en \
-output outputs/exp.txt \
-batch_size 1024 \
-gpu 0