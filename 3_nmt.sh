#! /bin/sh

# Step 2:
# Start training. Models are saved as `exp-model_acc_X_ppl_X_eX.pt`
python nmt/train.py \
-data dataset/jieba_50000 \
-save_model ckpts/adam_0.02 \
-gpuid 0 \
-learning_rate 0.02 \
-report_every 100