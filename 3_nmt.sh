#! /bin/sh

# Step 2:
# Start training. Models are saved as `exp-model_acc_X_ppl_X_eX.pt`
python nmt/train.py \
-data dataset/exp \
-save_model ckpts/demo-model \
-train_from ckpts/demo-model_acc_53.97_ppl_9.92_e10.pt \
-batch_size 128 \
-start_epoch 11 \
-gpuid 0 \
-learning_rate 0.002 \
-report_every 100