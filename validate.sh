#! /bin/sh

# 将`outputs/exp.txt`打包为`outputs/exp.sgm``
./tools/wrap_xml.pl zh ./raw_data/translation_validation_20170912/valid.en-zh.en.sgm exp < outputs/exp.txt > outputs/exp.sgm

# 将exp.sgm单字分割为exp.seg.sgm
./tools/chi_char_segment.pl -t xml < outputs/exp.sgm > outputs/exp.seg.sgm
# 将valid.en-zh.zh.sgm分割为valid.seg.sgm
./tools/chi_char_segment.pl -t xml < ./raw_data/translation_validation_20170912/valid.en-zh.zh.sgm > outputs/valid.seg.sgm

# 计算BLEU，输出至score/exp.bleu
./tools/mteval-v11b.pl -s ./raw_data/translation_validation_20170912/valid.en-zh.en.sgm -r outputs/valid.seg.sgm -t outputs/exp.seg.sgm -c > outputs/exp.bleu