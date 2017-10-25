#! /bin/sh

# Step 3:
# Output inferences.
python nmt/translate.py \
-model ckpts/demo-model_acc_51.94_ppl_12.01_e4.pt \
-src data/valid.en \
-output outputs/exp.txt \
-replace_unk \
-verbose
-gpu 0