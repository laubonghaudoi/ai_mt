#! /bin/sh

# 将纯文本`exp`转为`exp.sgm`
# exp 可更改为本次实验名
./tools/wrap_xml.pl zh ./outputs/valid.en-zh.en.sgm exp < outputs/exp.txt > outputs/exp.sgm

# 将exp.sgm分割为exp.seg.sgm
./tools/chi_char_segment.pl -t xml < outputs/exp.sgm > outputs/exp.seg.sgm
# 将valid.en-zh.zh.sgm分割为valid.en-zh.zh.seg.sgm
./tools/chi_char_segment.pl -t xml < outputs/valid.en-zh.zh.sgm > outputs/valid.en-zh.zh.seg.sgm

# 计算BLEU，输出至score/exp.bleu
./tools/mteval-v11b.pl -s outputs/valid.en-zh.en.sgm -r outputs/valid.en-zh.zh.seg.sgm -t outputs/exp.seg.sgm -c > score/exp.bleu
# 计算BLEU并打印
python ./tools/mt-score-main.py -rs outputs/valid.en-zh.zh.sgm -hs outputs/exp.sgm -ss outputs/valid.en-zh.en.sgm --id exp | tee score/exp.score