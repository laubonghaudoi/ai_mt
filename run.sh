#! /bin/sh

# 将纯文本`out1`转为sgm文件
./tools/wrap_xml.pl zh ./output/valid.en-zh.en.sgm DemoSystem < output/out1 > output/out1.sgm

# 将out1.sgm分割为out1.seg.sgm
./tools/chi_char_segment.pl -t xml < output/out1.sgm > output/out1.seg.sgm
# 将valid.en-zh.zh.sgm分割为valid.en-zh.zh.seg.sgm
./tools/chi_char_segment.pl -t xml < output/valid.en-zh.zh.sgm > output/valid.en-zh.zh.seg.sgm

# 计算BLEU
./tools/mteval-v11b.pl -s output/valid.en-zh.en.sgm -r output/valid.en-zh.zh.seg.sgm -t output/out1.seg.sgm -c > score/out1.bleu