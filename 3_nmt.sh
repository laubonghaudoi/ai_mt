#! /bin/sh

# Step 2:
# Start training. Models are saved as `exp-model_acc_X_ppl_X_eX.pt`
python nmt/train.py \
-data dataset/cross_2 \
-save_model ckpts/exp-model \
-train_from ckpts/demo-model_acc_51.94_ppl_12.01_e4.pt \
-start_epoch 5 \
-gpuid 0