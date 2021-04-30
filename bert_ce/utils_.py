#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: utils_.py

@time: 2019/04/9 10:30

@desc: 处理句子
"""

import re
import tensorflow as tf
import os
import pickle
import codecs

class SentenceProcessor(object):
    def __init__(self):
        self.sentence_index = 0

    @staticmethod
    def cut_sentence(sentence):
        """
        分句
        :arg
        sentence: string类型，一个需要分句的句子

        :return
        返回一个分好句的列表
        """
        sentence = re.sub('([。！？\?])([^”’])', r"\1\n\2", sentence)
        sentence = re.sub('(\.{6})([^”’])', r"\1\n\2", sentence)
        sentence = re.sub('(\…{2})([^”’])', r"\1\n\2", sentence)
        sentence = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', sentence)
        sentence = sentence.rstrip()

        return sentence.split("\n")

    def concat_sentences(self, sentences, max_seq_length):
        """
        把一个列表里的句子按照小于最大长度拼接为另一个句子，当某几个句子拼接
        达到max_seq_length长度的时候，把这个新的句子保存到新的列表当中。

        :arg
        sentences: list类型，一个要别拼接的句子列表
        max_seq_length: 拼接的新句子的最大长度

        :return
        一个新的拼接句子的列表，元素为string类型，即句子
        """
        # 分句后并从新拼接的句子
        new_sentences = []
        # 句子的index，是一个两层的列表，例如 [[0], [1, 2]], 列表内的每一个列表，
        # 表示来源于同一个句子，这里[1, 2]就表示是同一个句子被分割的两个句子
        sentences_index = []

        for sentence in sentences:
            sentence = self.clean_sentence(sentence)
            # 如果句子小于且等于最大长度的话，不进行处理
            if len(sentence) <= max_seq_length:
                new_sentences.append(sentence)
                sentences_index.append([self.sentence_index])
                self.sentence_index += 1

            # 如果句子大于最大长度就需要进行切割句子再拼接的操作了
            else:
                # 产生拼接句子列表（列表内每个句子小于最大长度）和同一个句子的index列表
                single_sentences, singe_index = self.concat_single_sentence(sentence, max_seq_length)
                new_sentences.extend(single_sentences)
                sentences_index.append(singe_index)

        # 当调用完此函数后，需要把sentence_index设为0，否则下次再次使用时候，将不会从0开始记录
        self.sentence_index = 0

        return new_sentences, sentences_index

    def concat_single_sentence(self, sentence, max_seq_length):
        """
        把一个句子分句为多个句子，把这些句子再拼接成若干个小于
        max_seq_length的句子

        :arg
        sentence: string类型，待分割的句子

        :return
        拼接后的句子列表和同一个句子的index列表
        """
        # 拼接后的句子列表
        single_sentences = []
        # 同一个句子的index列表
        singe_index = []
        tmp = ''
        # 分句， 注意此时sentence为list类型
        sentence = self.cut_sentence(sentence)
        for i, sent in enumerate(sentence):
            tmp = tmp + sent
            if len(tmp) > max_seq_length:
                pre = tmp[0: len(tmp) - len(sent)]
                if len(pre) >= 2:
                    single_sentences.append(pre)
                    singe_index.append(self.sentence_index)
                    self.sentence_index += 1
                tmp = sent

            # 当遍历到最后一个的时候，且tmp不为空字符串，就把tmp存入single_sentences中
            if i == len(sentence) - 1 and len(tmp) >= 2:
                single_sentences.append(tmp)
                singe_index.append(self.sentence_index)
                self.sentence_index += 1

        return single_sentences, singe_index

    @staticmethod
    def clean_sentence(sentence):
        sentence = sentence.strip()
        sentence = re.sub('\t| ', '', sentence)

        return sentence

def convert_single_lable_ids(example, label_list, max_seq_length, tokenizer, output_dir):
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
    with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as r:
        label_map = pickle.load(r)
    print('label_map:', label_map)
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

if __name__ == '__main__':
    test = ['四天三次!4月14日，马云第三次谈996，他表示看到了大家的质疑，但还是想说实话。“12315,996关键在于找到自己喜欢的事，真正的996不是简单加班，而是把时间用在学习和提升自己，爱觉不累，但企业不能不给钱。',
        '马云老师的一番言论，又在网上引起热议。',
        '我是',
        '你知不知道龙门石窟在哪个地方']
    sp = SentenceProcessor()
    a, b = sp.concat_sentences(test, 62)