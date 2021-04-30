#-*- coding:utf-8 -*-
"""
    desc:
    author:MeteorMan
    datetime:2021/4/28
"""
import tensorflow as tf
import codecs
import os
import pickle

def convert_single_example(example, label_list, max_seq_length, tokenizer, output_dir):
    """
    把一个example转换为feature，这一个过程会进行token的处理
    :arg
    ex_index: 是第几个example
    example: 单个InputExample的实例对像
    label_list: label的列表
    max_seq_length: 输入序列的最大的长度
    tokenizer: FullTokenizer的实例化对象
    output_dir: 模型存储路径
    mode: train/eval/test 模式
    """
    # label_list 映射到 int, 例如['']
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 检验label2id二进制文件是否存在
    if not tf.gfile.Exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
    else:
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as r:
            label_map = pickle.load(r)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')

    assert len(textlist) == len(labellist)

    tokens = []
    labels = []

    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        # 通过检验token的长度决定是否使用label_map中的标签
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    # 截断序列
    if len(tokens) >= max_seq_length - 1:
        # -2 的原因是因为序列需要加一个句首和句尾标志
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    # 把[CLS]，[SEP]加入到句首和句尾
    ntokens = []
    label_ids = []
    # 句子开始设置CLS 标志
    ntokens.append("[CLS]")
    # 因为只有一句输入
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        label_ids.append(label_map[labels[i]])

    ntokens.append("[SEP]")
    label_ids.append(label_map["[SEP]"])

    # 把token转化为id的形式
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    # 要注意的是input_mask是由1、0组成的，1表示真实的token，
    # 0表示填充的token，在这里并不是预训练的阶段，所以只能
    # 使用1来表示
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        label_ids.append(0)
        ntokens.append("**NULL**")

    return label_ids
