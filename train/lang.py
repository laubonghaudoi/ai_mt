SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    lang1_lines = open('prepare/t2t_data/train.%s' %
                       lang1, encoding='utf-8').read().strip().split('\n')
    lang2_lines = open('prepare/t2t_data/train.%s' %
                       lang2, encoding='utf-8').read().strip().split('\n')
    # Read the file and split into lines
    # lines = open('train/data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
    #    read().strip().split('\n')
    assert len(lang1_lines) == len(lang2_lines)
    n = len(lang1_lines)
    pairs = [[lang1_lines[l], lang2_lines[l]] for l in range(n)]
    # Split every line into pairs and normalize
    #pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, config):
    return len(p[0].split(' ')) < config.max_length and len(p[1].split(' ')) < config.max_length


def filterPairs(pairs, config):
    return [pair for pair in pairs if filterPair(pair, config)]


def prepareData(lang1, lang2, config):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, config.reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, config)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs
