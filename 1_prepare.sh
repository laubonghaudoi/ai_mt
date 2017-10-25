#! /bin/sh

# Step 1 (Manually): 
# Download the dataset and put the dataset in ../raw_data file

# Step 2:
# Create path for preprocessed data
DATA_DIR=./tokenized_jieba_data
TMP_DIR=./raw_data
mkdir -p $DATA_DIR

# Step 3:
# Unwrap xml for valid data and test data 
# Train data are not xml files thus need not unwrapping
python prepare/unwrap_xml.py $TMP_DIR/translation_validation_20170912/valid.en-zh.zh.sgm > $DATA_DIR/valid.en-zh.zh
python prepare/unwrap_xml.py $TMP_DIR/translation_validation_20170912/valid.en-zh.en.sgm > $DATA_DIR/valid.en-zh.en
python prepare/unwrap_xml.py test/ai_challenger_translation_test_a_20170923.sgm > $DATA_DIR/test_a.en

# Step 4:
# Chinese words segmentation
python prepare/jieba_cws.py $TMP_DIR/translation_train_20170912/train.zh > $DATA_DIR/train.zh
python prepare/jieba_cws.py $DATA_DIR/valid.en-zh.zh > $DATA_DIR/valid.zh

# Step 5:
# Tokenize and Lowercase English training data
chmod 777 prepare/tokenizer.perl
cat $TMP_DIR/translation_train_20170912/train.en | prepare/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/train.en
cat $DATA_DIR/valid.en-zh.en | prepare/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/valid.en
cat $DATA_DIR/test_a.en | prepare/tokenizer.perl -l en | tr A-Z a-z > $DATA_DIR/test_a.en

# Step 6:
# Bulid Dictionary
python prepare/build_dictionary.py $DATA_DIR/train.en
python prepare/build_dictionary.py $DATA_DIR/train.zh
src_vocab_size=50000
trg_vocab_size=50000
python prepare/generate_vocab_from_json.py $DATA_DIR/train.en.json ${src_vocab_size} > $DATA_DIR/vocab.en
python prepare/generate_vocab_from_json.py $DATA_DIR/train.zh.json ${trg_vocab_size} > $DATA_DIR/vocab.zh

# Step 7
# Remove temporary files
rm -r $DATA_DIR/train.*.json
rm -r $DATA_DIR/valid.en-zh.*