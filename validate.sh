#! /bin/sh

# 将纯文本`out1`转为`out1.sgm`
# ex1 可更改为本次实验名
./tools/wrap_xml.pl zh ./outputs/ex1/valid.en-zh.en.sgm ex1 < outputs/ex1/out1 > outputs/ex1/out1.sgm

# 将out1.sgm分割为out1.seg.sgm
./tools/chi_char_segment.pl -t xml < outputs/ex1/out1.sgm > outputs/ex1/out1.seg.sgm
# 将valid.en-zh.zh.sgm分割为valid.en-zh.zh.seg.sgm
./tools/chi_char_segment.pl -t xml < outputs/ex1/valid.en-zh.zh.sgm > outputs/ex1/valid.en-zh.zh.seg.sgm

# 计算BLEU，输出至score/out1.bleu
./tools/mteval-v11b.pl -s outputs/ex1/valid.en-zh.en.sgm -r outputs/ex1/valid.en-zh.zh.seg.sgm -t outputs/ex1/out1.seg.sgm -c > score/ex1.bleu
# 计算BLEU并打印
python ./tools/mt-score-main.py -rs outputs/ex1/valid.en-zh.zh.sgm -hs outputs/ex1/out1.sgm -ss outputs/ex1/valid.en-zh.en.sgm --id ex1 | tee score/ex1.score