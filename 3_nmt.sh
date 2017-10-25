#! /bin/sh

# Step 2:
# Start training. Models are saved as `exp-model_acc_X_ppl_X_eX.pt`
python nmt/train.py \
-data data/exp \
-save_model exp-model\
-train_from demo-model_acc_51.94_ppl_12.01_e4.pt\
-gpuid 0 \
-batch_size 64 \
-epochs 10 \
-optim adam \ 
-learning_rate 0.001