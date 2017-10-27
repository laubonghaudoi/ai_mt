#! /bin/sh

# Step 3:
# Output inferences.
python nmt/translate.py \
-model ckpts/demo-model_acc_53.97_ppl_9.92_e10.pt \
-src tokenized_jieba_data/valid.en \
-output outputs/exp.txt \
-batch_size 1024 \
-gpu 0