# 运行基线模型并提交测试

## 概述

本文档介绍如何完整运行基线模型并提交至官网，获得成绩。全过程分三步：数据预处理、训练模型、测试提交。

比赛官方验证脚本和基线模型地址为[AI_Challenger](https://github.com/AIChallenger/AI_Challenger)。经测试，其中的数据预处理代码，即`AI_Challenger/Baselines/translation_and_interpretation_baseline/train/prepare_data/*`可直接移植。但训练模型代码`AI_Challenger/Baselines/translation_and_interpretation_baseline/train/run.sh`有bug无法运行。因此，需要移植另一开源项目代码[fairseq-zh-en](https://github.com/twairball/fairseq-zh-en)

## 数据预处理

数据预处理流程全部运行于python 2.7和perl 5环境，依赖包有

- jieba 0.39

#### 步骤

1. 从官网下载训练集`ai_challenger_translation_train_20170912.zip`及验证集`ai_challenger_translation_validation_20170912.zip`。然后解压并复制如下
    ```bash
    /raw_data
        /ai_challenger_translation_train_20170912
            train.en
            train.zh
        /ai_challenger_translation_validation_20170912
            valid.en-zh.en.sgm
            valid.en-zh.zh.sgm
    /train
        ...
    ```
1. 预处理数据，运行以下命令
    ```bash
    cd prepare
    # 更改读写权限，防止出现Permission dennied错误
    chmod 777 prepare.sh
    chmod 777 ./prepare_data/tokenizer.perl
    # 预处理数据
    ./prepare.sh
    # 此脚本在i5 4300上运行时间约为10分钟。执行完毕后数据存放于`./prepare/t2t_data`
    ```
    `prepare.sh`原理解释如下：
    1. 运行`./train/prepare_data/unwrap_xml.py`去除验证集中xml代码，保存至`valid.en-zh.en`和`valid.en-zh.zh`
    1. 运行`./train/prepare_data/jieba_cws.py`将中文训练集和验证集分词并保存至`train.zh`和`valid.zh`
    1. 运行`./train/prepare_data/tokenizer.perl`将英文训练集和验证集tokenize并保存至`train.en`和`valid.en`
    1. 运行`./train/prepare_data/build_dictionary.py`建立中英文训练集词表并保存至`vocab.en`和`vocab.zh`
    1. 运行`./train/prepare_data/generate_vocab_from_json.py`去除训练集中低频词

## 训练模型


训练流程全部运行于python 3.6环境，依赖包有

- pytorch
- fairseq

安装过程见
[From Source](https://github.com/pytorch/pytorch#from-source)
[Requirements and Installation](https://github.com/facebookresearch/fairseq-py#requirements-and-installation)

#### 步骤

## 测试提交

本部分全部运行于python 2.7和perl 5环境。

利用`valid.en`输入，现有文件

- 翻译后纯文本文件 `./output/out1`
- 中文参考翻译文件`./output/valid.en-zh.zh.sgm`
- 英文源文件`./output/valid.en-zh.en.sgm`

现须输出可提交文件`./output/out1.sgm`和分数`./output/out1.score`

#### 步骤

直接运行
```bash
chmod 777 run.sh
./run.sh
```