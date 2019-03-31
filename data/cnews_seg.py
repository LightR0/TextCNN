# coding: utf-8

import jieba
import warnings
from collections import Counter
warnings.filterwarnings("ignore")
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# 判断字符串是否是小数或者数字
def check_number(string):
    str1 = str(string)
    if str1.count('.') > 1: #判断小数点是不是大于1
        return False
    elif str1.isdigit():
        return True #判断是不是整数
    else:
        new_str = str1.split('.')#按小数点分割字符
        frist_num = new_str[0] #取分割完之后这个list的第一个元素
        if frist_num.count('-') > 1:#判断负号的格数，如果大于1就是非法的
            return False
        else:
            frist_num = frist_num.replace('-','')#把负号替换成空
        if frist_num.isdigit() and new_str[1].isdigit():
        #如果小数点两边都是整数的话，那么就是一个小数
            return True
        else:
            return False

# load stopwords
def load_stopwords(filename):
    stopwords = []
    with open(filename, 'r') as f:
        stopwords.extend([line.strip() for line in f])
    return stopwords


# read file and segment    
def read_file_seg(filename, save_file):
    contents, labels = [], []
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            if count % 5000 == 0:
                print 'finished: ', count
            count += 1
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append([i for i in jieba.cut(content.strip()) if len(i) > 1 and not check_number(i)])
                    labels.append(label)
            except:
                pass
    savefile = open(save_file, 'w')
    for i in range(len(contents)):
        savefile.write(labels[i] + '\t' + ",".join(contents[i]) + '\n')
    return contents, labels


# build vocab
def build_vocab(contents, _vocab_dir, vocab_size=3000):
    stop_words = load_stopwords('cnews/stopwords.txt')
    all_data = []
    count = 0
    for content in contents:
        if count % 100 == 0:
            print 'finished: ', count
        count += 1
        all_data.extend([i for i in content if i not in stop_words])

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(_vocab_dir, mode='w').write('\n'.join(words) + '\n')


if __name__ == '__main__':
    train_contents, _ = read_file_seg('cnews/cnews.train.txt', 'cnews/cnews.train.seg.txt')
    print 'finished train data segment'
    test_contents, _ = read_file_seg('cnews/cnews.test.txt', 'cnews/cnews.test.seg.txt')
    print 'finished test data segment'
    val_contents, _ = read_file_seg('cnews/cnews.val.txt', 'cnews/cnews.val.seg.txt')
    print 'finished val data segment'
    build_vocab(train_contents, 'cnews/cnews.vocab.seg.txt')
