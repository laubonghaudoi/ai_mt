# 运行基线模型并提交测试

## 概述

本文档介绍如何完整运行基线模型并提交至官网，获得成绩。全过程分三步：数据预处理、训练模型、测试提交。

比赛官方验证脚本和基线模型地址为[AI_Challenger](https://github.com/AIChallenger/AI_Challenger)。经测试，其中的数据预处理代码，即`AI_Challenger/Baselines/translation_and_interpretation_baseline/train/prepare_data/*`可直接移植，数据提交代码亦可用。但训练模型代码`AI_Challenger/Baselines/translation_and_interpretation_baseline/train/run.sh`有bug无法运行。因此，训练代码使用[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

## 数据预处理

数据预处理流程全部运行于python 2.7和perl 5环境，依赖包有

- jieba 0.39

### 步骤

1. 从官网下载训练集`ai_challenger_translation_train_20170912.zip`及验证集`ai_challenger_translation_validation_20170912.zip`。然后解压并复制至目录如下
    ```bash
    /raw_data
        /ai_challenger_translation_train_20170912
            train.en
            train.zh
        /ai_challenger_translation_validation_20170912
            valid.en-zh.en.sgm
            valid.en-zh.zh.sgm
    /prepare
        ...
    /outputs
        ...
    /score
        ...
    /tools
        ...
    ```
1. 预处理数据，运行以下命令
    ```
    # 更改读写权限，防止出现Permission dennied错误
    chmod 777 prepare.sh
    # 预处理数据
    # 此脚本使用单线程运行，在i5 4300上运行时间约为10分钟。执行完毕后数据存放于`./prepare/t2t_data`
    ./prepare.sh
    ```
    `prepare.sh`原理解释如下：
    1. 运行`./prepare/unwrap_xml.py`去除验证集、测试集中xml代码，保存至`./tokenized_jieba_data/valid.en-zh.en`、`./tokenized_jieba_data/valid.en-zh.zh`和`./tokenized_jieba_data/test_a.en`
    1. 运行`./prepare/jieba_cws.py`将中文训练集和验证集分词并保存至`./tokenized_jieba_data/train.zh`和`./tokenized_jieba_data/valid.zh`
    1. 运行`./prepare/tokenizer.perl`将英文训练集和验证集tokenize并保存至`./tokenized_jieba_data/train.en`和`./tokenized_jieba_data/valid.en`
    1. 运行`./prepare/build_dictionary.py`建立中英文训练集词表并保存至`./tokenized_jieba_data/vocab.en`和`./tokenized_jieba_data/vocab.zh`
    1. 运行`./prepare/generate_vocab_from_json.py`去除训练集中低频词
1. 预处理完毕后，当前路径文件结构如下：
    ```bash
    prepare/
        ...
    raw_data/
        ...
    tokenized_jieba_data/
        # 以下为预处理完毕后文件，可直接输入模型训练
        test_a.en   # tokenize后英文测试集
        train.en    # tokenize后英文训练集
        train.zh    # 分词后中文训练集
        valid.en    # tokenize后英文验证集
        valid.zh    # 分词后中文验证集
        vocab.en    # 英文训练集词表
        vocab.zh    # 中文训练集词表
    ```

## 训练模型

训练流程全部运行于python 3.6环境，依赖包有

- [pytorch](https://github.com/pytorch/pytorch#from-source)
- [torchtext](https://github.com/pytorch/text) == 0.1.1
- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

### 步骤

#### 1. 构建Dataset对象

```bash
# 读取预处理后纯文本文件，构造Dataset对象并保存至`dataset/`
chmod 777 2_preprocess.sh
# 注意，此步骤有可能爆内存，可将`train.en`削减一半以避免
./2_preprocess.sh
```

`2_preprocess.sh`中可调参数如下

```bash
[-config CONFIG]
[-src_type SRC_TYPE]
[-src_img_dir SRC_IMG_DIR]
-train_src TRAIN_SRC
-train_tgt TRAIN_TGT
-valid_src VALID_SRC
-valid_tgt VALID_TGT
-save_data SAVE_DATA
[-src_vocab_size SRC_VOCAB_SIZE]
[-tgt_vocab_size TGT_VOCAB_SIZE]
[-src_vocab SRC_VOCAB]
[-tgt_vocab TGT_VOCAB]
[-features_vocabs_prefix FEATURES_VOCABS_PREFIX]
[-src_seq_length SRC_SEQ_LENGTH]
[-src_seq_length_trunc SRC_SEQ_LENGTH_TRUNC]
[-tgt_seq_length TGT_SEQ_LENGTH]
[-tgt_seq_length_trunc TGT_SEQ_LENGTH_TRUNC]
[-shuffle SHUFFLE]
[-seed SEED]
[-lower]
[-report_every REPORT_EVERY]
```

#### 2. 开始训练

```bash
# 使用默认参数训练
chmod 777 3_nmt.sh
./3_nmt.sh
```

`3_nmt.sh`中训练参数如下，可自行修改

```bash
-data DATA
[-save_model SAVE_MODEL='model']    # Model filename (the model will be saved as <save_model>_epochN_PPL.pt where PPL is the validation perplexity
[-train_from_state_dict TRAIN_FROM_STATE_DICT]  # If training from a checkpoint then this is the path to the pretrained model's state_dict.
[-train_from TRAIN_FROM=''] # If training from a checkpoint then this is the path to the pretrained model's state_dict.
[-layers LAYERS=-1] # Number of layers in enc/dec=2.
[-rnn_size RNN_SIZE=500]    # Size of LSTM hidden states
[-word_vec_size WORD_VEC_SIZE=-1]   # Word embedding for both=500
[-feature_vec_size FEATURE_VEC_SIZE]    # If specified, feature embedding sizes will be set to this. Otherwise, feat_vec_exponent will be used.
[-input_feed INPUT_FEED=1]  # Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.
[-rnn_type {LSTM,GRU,SRU}]  # The gate type to use in the RNNs
[-brnn] # Deprecated, use `encoder_type'
[-brnn_merge BRNN_MERGE='concat']    # Merge action for the bidir hidden states
[-copy_attn]
[-coverage_attn]
[-encoder_layer ENCODER_LAYER]
[-decoder_layer DECODER_LAYER]
[-context_gate {source,target,both}]
[-encoder_type ENCODER_TYPE]
[-batch_size BATCH_SIZE]
[-max_generator_batches MAX_GENERATOR_BATCHES]
[-epochs EPOCHS]
[-start_epoch START_EPOCH]
[-param_init PARAM_INIT]
[-optim OPTIM]
[-max_grad_norm MAX_GRAD_NORM]
[-dropout DROPOUT]
[-position_encoding]
[-share_decoder_embeddings]
[-curriculum]
[-extra_shuffle]
[-truncated_decoder TRUNCATED_DECODER]
[-learning_rate LEARNING_RATE]
[-learning_rate_decay LEARNING_RATE_DECAY]
[-start_decay_at START_DECAY_AT]
[-start_checkpoint_at START_CHECKPOINT_AT]
[-decay_method DECAY_METHOD]
[-warmup_steps WARMUP_STEPS]
[-pre_word_vecs_enc PRE_WORD_VECS_ENC]
[-pre_word_vecs_dec PRE_WORD_VECS_DEC]
[-gpus GPUS [GPUS ...]]
[-log_interval LOG_INTERVAL]
[-log_server LOG_SERVER]
[-experiment_name EXPERIMENT_NAME]
[-seed SEED]
```

#### 3. 交叉训练与继续训练

训练过程中的模型会自动保存至`ckpts/`中，若要继续训练，可修改`3_nmt.sh`中`-train_from`参数，再重新运行此脚本即可

#### 4. 输出预测

训练完毕后，得到保存的模型`ckpts/demo-model_acc_x_ppl_x_e4x.pt`。修改`4_output.sh`中的以下参数：

```bash
-model # 要预测的模型名，demo-model_acc_x_ppl_x_e4x.pt
-src # 源文本，tokenized_jieba_data/test_a.en或tokenized_jieba_data/valid.en
-output # 预测文本输出路径，outputs/exp.txt
```

运行以下命令，即可输出预测文件

```
chmod 777 ./4_output.sh
./4_output.sh
```

## 测试提交

本部分全部运行于python 2.7和perl 5环境。

现有上一步中得到的预测文本文件`outputs/exp.txt`。为作示范，现有一次实验结果`outputs/`，内含文件：

```bash
outputs/ex1/
    out1    # 翻译后纯文本文件
    valid.en-zh.zh.sgm  # 中文参考翻译文件（标签）
    valid.en-zh.en.sgm  # 英文源文件
```

现须输出可提交文件`./outputs/ex1/out1.sgm`和分数`./score/out1.score`

### 步骤

直接运行以下代码，即可输出测试结果`./score/out1.bleu`，提交文件即为`out1.sgm`

```bash
chmod 777 validate.sh
./validate.sh
```

`validate.sh`原理解释如下：

1. 运行`./tools/wrap_xml.pl`将纯文本翻译输出`out1`转为`out1.sgm`
1. 运行`./tools/chi_char_segment.pl`将`out1.sgm`和`valid.en-zh.zh.sgm`分割为`out1.seg.sgm`和`valid.en-zh.zh.seg.sgm`，用于下一步计算BLEU
1. 运行`./tools/mteval-v11b.pl`计算BLEU，输出结果`./score/out1.bleu`

注意以上代码仅作示范，在正式实验时需修改`validate.sh`中的输出路径和实验名再运行