#-*- coding:utf-8 -*-
"""
    desc:
    将

    厦 B-LOC
    门 I-LOC
    与 O
    金 B-LOC
    门 I-LOC

    转换成
    厦 门 与 金 门-seq-B-LOC I-LOC O B-LOC I-LOC

    author:MeteorMan
    datetime:2021/4/27
"""
import os
import jieba
import tqdm

write_path = '../china-people-daily-data'

def convert(file):
    full_name = '../china-people-daily-data-lie/{}'.format(file)
    lines = []
    for line in open(full_name, 'r', encoding='utf-8'):
        if line.strip() == '':
            line = '|&|'
        lines.append(line.strip())

    words = []
    labels = []
    for item in lines:
        if item == '|&|':
            words.append(item)
            labels.append(item)
        else:
            words.append(item.split(' ')[0])
            labels.append(item.split(' ')[1])

    words = ''.join(words).split('|&|')
    labels = ''.join(labels).split('|&|')
    new_lines = []
    #注意要将标签信息更新
    label_item = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O']
    for lai in label_item:
        jieba.add_word(lai)
    for i in tqdm.tqdm(range(len(words)), desc='转换{}'.format(file)):
        if words[i] == '':
            continue
        sent = ' '.join(list(words[i]))
        label = ''
        for seg in jieba.cut(labels[i]):
            if seg not in label_item:
                seg = ' '.join(list(seg))
                label += seg
            else:
                label += ' ' + seg + ' '
        label = label.strip().replace('  ', ' ')
        # print(sent)
        # print(labels[i])
        # print(label)
        assert len(sent.split(' ')) == len(label.split(' '))
        new_line = sent + '-seq-' + label + '\n'
        # print(new_line)
        new_lines.append(new_line)

    print('数量：', len(new_lines))
    print('Example：', new_lines[0])

    write_file = os.path.join(write_path, file)
    with open(write_file, 'w', encoding='utf-8') as f:
        for line_item in new_lines:
            f.write(line_item)
        f.close()

if __name__ == '__main__':
    files = ['train.txt', 'dev.txt', 'test.txt']
    for file in files:
        convert(file)


