#-*- coding:utf-8 -*-
"""
    desc:
    将

    厦 门 与 金 门-seq-B-LOC I-LOC O B-LOC I-LOC

    转换成

    厦 B-LOC
    门 I-LOC
    与 O
    金 B-LOC
    门 I-LOC

    author:MeteorMan
    datetime:2021/4/29
"""
import os
import jieba
import tqdm

input_path = '../msra_data'
write_path = '../msra_data_lie'

def convert(file):
    full_file = os.path.join(input_path, file)
    texts = []
    labels = []
    for line in open(full_file, 'r', encoding='utf-8'):
        line = line.strip().split('-seq-')
        word = line[0].split(' ')
        label = line[1].split(' ')
        texts.append(word)
        labels.append(label)
    assert len(texts) == len(labels)
    lines = list(zip(texts, labels))
    new_lines = []
    for item in lines:
        # item = list(zip(item))
        # print(item)
        # print(len(item[0]), len(item[1]))
        text = item[0]
        label = item[1]
        for i in range(len(text)):
            char = text[i]
            lab = label[i]
            written_text = char + ' ' + lab + '\n'
            new_lines.append(written_text)
        new_lines.append('\n')
    print(len(new_lines))
    print(new_lines[0])

    if not os.path.exists(write_path):
        os.mkdir(write_path)

    write_file = os.path.join(write_path, file)
    with open(write_file, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)
        f.close()


if __name__ == '__main__':
    files = ['train.txt', 'dev.txt', 'test.txt']
    for file in files:
        convert(file)


