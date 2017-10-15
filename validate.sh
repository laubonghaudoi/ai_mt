#! /bin/sh

# 将纯文本`out1`转为`out1.sgm`
# ex1 可更改为本次实验名
./tools/wrap_xml.pl zh ./outputs/ex1/valid.en-zh.en.sgm ex1 < outputs/ex1/out1 > outputs/ex1/out1.sgm

# 将out1.sgm分割为out1.seg.sgm
./tools/chi_char_segment.pl -t xml < outputs/ex1/out1.sgm > outputs/ex1/out1.seg.sgm
# 将valid.en-zh.zh.sgm分割为valid.en-zh.zh.seg.sgm
./tools/chi_char_segment.pl -t xml < outputs/ex1/valid.en-zh.zh.sgm > outputs/ex1/valid.en-zh.zh.seg.sgm

# 计算BLEU
./tools/mteval-v11b.pl -s outputs/ex1/valid.en-zh.en.sgm -r outputs/ex1/valid.en-zh.zh.seg.sgm -t outputs/ex1/out1.seg.sgm -c > score/out1.bleu