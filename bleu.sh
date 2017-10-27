#! /bin/sh

# 计算BLEU并打印
./tools/wrap_xml.pl zh ./raw_data/translation_validation_20170912/valid.en-zh.en.sgm exp < outputs/exp.txt > outputs/exp.sgm
python ./tools/mt-score-main.py -rs ./raw_data/translation_validation_20170912/valid.en-zh.zh.sgm -hs outputs/exp.sgm -ss ./raw_data/translation_validation_20170912/valid.en-zh.en.sgm --id exp | tee score/exp.score