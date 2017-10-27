lines = open('train.en', encoding='utf-8').readlines()

n = len(lines)

l = int(n/2)

out = open('train_1.en', 'w', encoding='utf-8')
out.writelines(lines[:l])
out.close()
out = open('train_2.en', 'w', encoding='utf-8')
out.writelines(lines[l:])
out.close()

lines = open('train.zh', encoding='utf-8').readlines()

assert len(lines) == n

out = open('train_1.zh', 'w', encoding='utf-8')
out.writelines(lines[:l])
out.close()
out = open('train_2.zh', 'w', encoding='utf-8')
out.writelines(lines[l:])
out.close()
