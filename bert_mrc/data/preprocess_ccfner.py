import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from utils import json_to_text, save_json, save_pickle, load_pickle
from utils import max_len

len_treshold = max_len - 2  # 留出两个位置给CLS 和 SEP

'''
    数据预处理包含三步：
    1.排查出训练集和测试集里的特殊字符       
        1.1 训练集特殊字符=训练集字符-额外自定义字符-label中的字符 
        1.2 测试集特殊字符=测试集字符-额外自定义字符-训练label里的字符
    2.去除训练集和测试集里的特殊字符（可选）
        2.1 去除训练集特殊字符时，要注意对应改变label的起始与结束位置
        2.2 去除测试集特殊字符时，要将去掉的字符位置保存起来，以便后续还原  即[[0,0],[1,1],....]
    3.将数据处理成json格式
        3.1 要处理文本长度过长问题，处理测试集文本长度时，要将哪些是拆分开的保存起来  [[1,-1],[1,0],[2,1],[3,2]..]
        3.2 将数据保存成json格式
'''


# 1.1排查出训练集里的特殊字符
def cal_train_additional_chars(train_data_path, train_label_path, save_path):
    train_data_file_names = os.listdir(train_data_path)
    lengths = len(train_data_file_names)
    train_data_additional_chars = set()
    label_additional_chars = set()

    extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
    for index in range(lengths):
        train_data_dir = os.path.join(train_data_path, str(index) + '.txt')
        train_label_dir = os.path.join(train_label_path, str(index) + '.csv')
        with open(train_data_dir, 'r', encoding='utf-8') as f1:
            lines_text = f1.readlines()
            raw_text = ''
            for line_text in lines_text:
                raw_text += line_text
            train_data_additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(raw_text)))

            train_label = pd.read_csv(train_label_dir)
            for row_index, row in train_label.iterrows():
                assert len(row) == 5
                ID = row['ID']
                Category = row['Category']
                Pos_b = row['Pos_b']
                Pos_e = row['Pos_e']
                Privacy = str(row['Privacy'])
                label_additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(Privacy)))

    additional_chars = train_data_additional_chars.difference(label_additional_chars)  # 去掉标签里含有的特殊字符
    additional_chars = additional_chars.difference(extra_chars)  # 去掉额外的一些标点符号
    save_pickle(additional_chars, save_path)  # 保存成pickle形式
    additional_chars = load_pickle(save_path)
    return additional_chars, train_data_additional_chars, label_additional_chars


# 1.2排查出测试集里的特殊字符
def cal_test_additional_chars(test_data_path, label_additional_chars, test_save_path, ):
    test_data_file_names = os.listdir(test_data_path)
    lengths = len(test_data_file_names)
    test_data_additional_chars = set()

    # new_extra_chars = set("／﹒–é/▲‧♥♡∩×『２〉×.è◆……①＆")

    extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】／……﹒–")
    for index in range(lengths):
        test_data_dir = os.path.join(test_data_path, str(index) + '.txt')

        with open(test_data_dir, 'r', encoding='utf-8') as f1:
            lines_text = f1.readlines()
            raw_text = ''
            for line_text in lines_text:
                raw_text += line_text
            test_data_additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(raw_text)))

    additional_chars = test_data_additional_chars.difference(label_additional_chars)  # 去掉标签里含有的特殊字符
    additional_chars = additional_chars.difference(extra_chars)  # 去掉额外的一些标点符号
    # additional_chars = additional_chars.difference(new_extra_chars)  # 去掉额外的一些标点符号
    save_pickle(additional_chars, test_save_path)  # 保存成pickle形式
    additional_chars = load_pickle(test_save_path)
    return additional_chars, test_data_additional_chars, label_additional_chars


train_data_path = '../data/train/data'
train_label_path = '../data/train/label'
test_data_path = '../data/test'
additional_chars_train_save_path = './train_additional_chars.pkl'
additional_chars_test_save_path = './test_additional_chars.pkl'

train_additional_chars, train_data_additional_chars, label_additional_chars = cal_train_additional_chars(
    train_data_path=train_data_path, train_label_path=train_label_path,
    save_path=additional_chars_train_save_path)

test_additional_chars, test_data_additional_chars, label_additional_chars = cal_test_additional_chars(test_data_path,
                                                                                                      label_additional_chars,
                                                                                                      additional_chars_test_save_path)

train_additional_chars = load_pickle(additional_chars_train_save_path)  # 长度：104
test_additional_chars = load_pickle(additional_chars_test_save_path)  # 长度：231

print(test_additional_chars)


# 2.1删除训练集里的特殊字符，并且处理好数据保存起来
def delete_train_special_chars(train_data_path, train_label_path, train_additional_chars, process_data_path,
                               process_label_path):
    train_data_file_names = os.listdir(train_data_path)
    lengths = len(train_data_file_names)
    delete_info = []
    for index in range(lengths):
        train_data_dir = os.path.join(train_data_path, str(index) + '.txt')
        train_label_dir = os.path.join(train_label_path, str(index) + '.csv')

        with open(train_data_dir, 'r', encoding='utf-8') as f1:
            lines_text = f1.readlines()
            raw_text = ''
            for line_text in lines_text:
                raw_text += line_text

        text = ''
        delete_info_single = []
        count = 0
        for char_index, char in enumerate(raw_text):
            if char not in train_additional_chars:
                delete_info_single.append([char_index, count])
                text += char
                count += 1
            else:
                delete_info_single.append([char_index, -1])

        process_data_dir = os.path.join(process_data_path, str(index) + '.txt')
        with open(process_data_dir, 'w', encoding='utf-8') as f2:
            f2.write(text)

        process_label_list = []
        train_label = pd.read_csv(train_label_dir)
        for row_index, row in train_label.iterrows():
            assert len(row) == 5
            ID = row['ID']
            Category = row['Category']
            Pos_b = row['Pos_b']
            Pos_e = row['Pos_e']
            Privacy = str(row['Privacy'])

            processed_start = delete_info_single[Pos_b][1]
            processed_end = delete_info_single[Pos_e][1]
            processed_Privacy = text[processed_start:processed_end + 1]

            assert processed_Privacy == Privacy

            process_label_list.append([ID, Category, processed_start, processed_end, processed_Privacy])

        process_label_pd = pd.DataFrame(process_label_list, columns=['ID', 'Category', 'Pos_b', 'Pos_e', 'Privacy'])
        process_label_dir = os.path.join(process_label_path, str(index) + '.csv')
        process_label_pd.to_csv(process_label_dir, encoding='utf-8', index=None)
        print(index)
        delete_info.append(delete_info_single)

    return delete_info, len(delete_info)


raw_train_data_path = '../data/train/data'
raw_train_label_path = '../data/train/label'
process_data_path = '../data/train/data_delete_chars'
process_label_path = '../data/train/label_delete_chars'
delete_info, len_delete_info = delete_train_special_chars(raw_train_data_path,
                                                          raw_train_label_path,
                                                          train_additional_chars,
                                                          process_data_path,
                                                          process_label_path)


# 2.2删除测试集里的特殊字符，并且处理好数据保存起来
def delete_test_special_chars(test_data_path, test_additional_chars, process_test_data_path,
                              test_delete_info_save_path):
    test_data_file_names = os.listdir(test_data_path)
    lengths = len(test_data_file_names)
    test_delete_info = []
    for index in range(lengths):
        test_data_dir = os.path.join(test_data_path, str(index) + '.txt')

        with open(test_data_dir, 'r', encoding='utf-8') as f1:
            lines_text = f1.readlines()
            raw_text = ''
            for line_text in lines_text:
                raw_text += line_text
        text = ''
        test_delete_info_single = []
        count = 0
        for char_index, char in enumerate(raw_text):
            if char not in test_additional_chars:
                test_delete_info_single.append([char_index, count])
                text += char
                count += 1
            else:
                test_delete_info_single.append([char_index, -1])
        print(index)
        test_process_data_dir = os.path.join(process_test_data_path, str(index) + '.txt')
        with open(test_process_data_dir, 'w', encoding='utf-8') as f2:
            f2.write(text)

        # print(index)

        test_delete_info.append(test_delete_info_single)
    save_pickle(test_delete_info, test_delete_info_save_path)  # 将删除的字符列表存入pkl文件
    return test_delete_info, len(test_delete_info)


raw_test_data_path = '../data/test'
process_test_data_path = '../data/test_data_delete_chars'
test_delete_info_save_path = './version_3_char/test_delete_info.pkl'
test_delete_info, len_test_delete_info = delete_test_special_chars(raw_test_data_path,
                                                                   test_additional_chars,
                                                                   process_test_data_path,
                                                                   test_delete_info_save_path)
#
# 3.将除掉特殊字符的文本写成json格式，包含了处理长文本
train_data_path = '../data/train/data_delete_chars'
train_label_path = '../data/train/label_delete_chars'
test_data_path = '../data/test_data_delete_chars'
json_save_path = './version_3_char/train.json'
test_json_save_path = './version_3_char/test.json'
sentences_seg_path = './version_3_char/test_sentences_seg.pkl'


# delete_info_path = './version_3_char/test_delete_info.pkl'
# test_delete_info = load_pickle(delete_info_path)


# 处理长文本，尽量用什么切开要考虑（）
def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    # new_sentence = []
    # sen = []
    # for i in sentence:
    #     if i in ['。', '！', '？', '?'] and len(sen) != 0:
    #         sen.append(i)
    #         new_sentence.append("".join(sen))
    #         sen = []
    #         continue
    #     sen.append(i)
    # if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
    #     new_sentence.append("".join(sen))
    # if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
    new_sentence = []
    sen = []
    for i in sentence:
        if i.split(' ')[0] in ['，', ',', '。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence


def write_json_label(json_d, Category, Privacy, start, end):
    if Category in json_d['label']:
        if Privacy in json_d['label'][Category]:
            json_d['label'][Category][Privacy].append([start, end])
        else:
            json_d['label'][Category][Privacy] = [[start, end]]
    else:
        json_d['label'][Category] = {}
        json_d['label'][Category][Privacy] = [[start, end]]


# 3.1 写成json格式保存
def write_data_json(train_data_path, train_label_path, json_save_path):
    train_data_file_names = os.listdir(train_data_path)
    lengths = len(train_data_file_names)
    all_sentences = []
    all_sentences_len = []
    for index in range(lengths):
        json_d = {}
        train_data_dir = os.path.join(train_data_path, str(index) + '.txt')
        train_label_dir = os.path.join(train_label_path, str(index) + '.csv')
        with open(train_data_dir, 'r', encoding='utf-8') as f1:
            lines_text = f1.readlines()
            raw_text = ''
            for line_text in lines_text:
                raw_text += line_text
            len_raw_text = len(raw_text)

            all_sentences_len.append(len_raw_text)

            train_label = pd.read_csv(train_label_dir)

            if len_raw_text < len_treshold:
                json_d['id'] = index
                json_d['text'] = raw_text
                json_d['label'] = {}

                for row_index, row in train_label.iterrows():
                    assert len(row) == 5
                    ID, Category, Pos_b, Pos_e, Privacy = row['ID'], row['Category'], row['Pos_b'], row['Pos_e'], row[
                        'Privacy']
                    start, end = int(Pos_b), int(Pos_e)

                    assert raw_text[start:end + 1] == str(Privacy)  # 验证处理的样本是否正确

                    write_json_label(json_d, Category, Privacy, start, end)
                all_sentences.append(json_d)
            else:
                print('长度超过128的训练句子索引：', index)
                sentence_list = _cut(raw_text)  # 切分句子
                text_agg = ''
                len_text_agg_sum = 0  # 存的是之前合并了多少长度，以便后续更改start和end
                count_label = 0  # 用于验证写入的标签数量是否完整
                for sentence in sentence_list:
                    if len(text_agg) + len(sentence) < len_treshold:
                        text_agg += sentence
                    else:
                        if text_agg == '':
                            text_agg = sentence
                            continue
                        json_d['id'] = index
                        json_d['text'] = text_agg
                        json_d['label'] = {}
                        for row_index, row in train_label.iterrows():
                            assert len(row) == 5
                            ID, Category, Pos_b, Pos_e, Privacy = row['ID'], row['Category'], row['Pos_b'], row[
                                'Pos_e'], row['Privacy']
                            start, end = int(Pos_b), int(Pos_e)

                            assert raw_text[start:end + 1] == str(Privacy)  # 验证处理的样本是否正确

                            if str(Privacy) in text_agg and start >= len_text_agg_sum and end < len_text_agg_sum + len(
                                    text_agg):
                                assert str(Privacy) == text_agg[start - len_text_agg_sum: end - len_text_agg_sum + 1]
                                if end - len_text_agg_sum + 1 > len_treshold:
                                    print('标签长度超了的索引：', index)
                                else:
                                    write_json_label(json_d, Category, Privacy, start - len_text_agg_sum,
                                                     end - len_text_agg_sum)
                                count_label += 1

                        len_text_agg_sum += len(text_agg)

                        if len(json_d['label']) > 0:
                            all_sentences.append(json_d)
                        text_agg = sentence
                        json_d = {}
                # 加回最后一个句子
                json_d['id'] = index
                json_d['text'] = text_agg
                json_d['label'] = {}
                for row_index, row in train_label.iterrows():
                    assert len(row) == 5
                    ID, Category, Pos_b, Pos_e, Privacy = row['ID'], row['Category'], row['Pos_b'], row['Pos_e'], row[
                        'Privacy']
                    start, end = int(Pos_b), int(Pos_e)
                    assert raw_text[start:end + 1] == str(Privacy)  # 验证处理的样本是否正确
                    if str(Privacy) in text_agg and start >= len_text_agg_sum and end < len_text_agg_sum + len(
                            text_agg):
                        assert Privacy == text_agg[start - len_text_agg_sum: end - len_text_agg_sum + 1]

                        if end - len_text_agg_sum + 1 > len_treshold:
                            print('标签长度超了的索引：', index)
                        else:
                            write_json_label(json_d, Category, Privacy, start - len_text_agg_sum,
                                             end - len_text_agg_sum)
                        count_label += 1

                if len(json_d['label']) > 0:
                    all_sentences.append(json_d)
                # 用来验证写入是否正确
                assert len_text_agg_sum + len(text_agg) == len(raw_text)

                if count_label != row_index + 1:
                    print('标签被切开的索引有：', index)

    json_to_text(json_save_path, all_sentences)
    return all_sentences, all_sentences_len


def write_test_data_json(test_data_path, test_json_save_path, sentences_seg_path):
    test_data_file_names = os.listdir(test_data_path)
    lengths = len(test_data_file_names)
    all_sentences = []
    all_sentences_len = []
    all_sentences_seg = []
    all_sentences_delete_index = []
    for index in range(lengths):
        json_d = {}
        count_seg = 0  # 切割次数计数

        test_data_dir = os.path.join(test_data_path, str(index) + '.txt')
        with open(test_data_dir, 'r', encoding='utf-8') as f1:

            lines_text = f1.readlines()
            raw_text = ''
            for line_text in lines_text:
                raw_text += line_text
            len_raw_text = len(raw_text)
            all_sentences_len.append(len_raw_text)

            if len_raw_text < len_treshold:
                json_d['id'] = index
                json_d['text'] = raw_text  # 测试集的字符先不去
                all_sentences.append(json_d)
                count_seg += 1
                all_sentences_seg.append((count_seg, -1))
            else:
                print('长度超过128的测试句子索引：', index)
                sentence_list = _cut(raw_text)  # 切分句子
                text_agg = ''
                len_text_agg_sum = 0  # 存的是之前合并了多少长度，以便后续更改start和end
                for sentence in sentence_list:
                    if len(text_agg) + len(sentence) < len_treshold:
                        text_agg += sentence
                    else:
                        if text_agg == '':
                            text_agg = sentence
                            continue
                        json_d['id'] = index
                        json_d['text'] = text_agg
                        all_sentences.append(json_d)
                        count_seg += 1
                        all_sentences_seg.append((count_seg, len_text_agg_sum))
                        len_text_agg_sum += len(text_agg)
                        text_agg = sentence
                        json_d = {}
                json_d['id'] = index
                json_d['text'] = text_agg
                if len(text_agg) > 0:
                    count_seg += 1
                    all_sentences.append(json_d)
                    all_sentences_seg.append((count_seg, len_text_agg_sum))
    json_to_text(test_json_save_path, all_sentences)
    save_pickle(all_sentences_seg, sentences_seg_path)
    return all_sentences, all_sentences_len, all_sentences_seg, all_sentences_delete_index


all_sentences_train, all_sentences_train_len = write_data_json(
    train_data_path=train_data_path,
    train_label_path=train_label_path,
    json_save_path=json_save_path)
all_sentences_test, all_sentences_test_len, all_sentences_seg, all_sentences_delete_index = write_test_data_json(
    test_data_path=test_data_path,
    test_json_save_path=test_json_save_path,
    sentences_seg_path=sentences_seg_path)


def plot_len(sentences_len):
    a = 0
    b = 0
    c = 0
    for index, sentence_len in enumerate(sentences_len):
        if sentence_len > 512:
            c += 1
        elif sentence_len > 256:
            b += 1
        else:
            a += 1
    print('长度在【0-256】之间有:{0}, 长度【257-512】之间有:{1}, 长度大于512之间有:{2}, '.format(a, b, c))
    print('最长文本长度为：{0}, 最短文本长度为：{1}'.format(max(sentences_len), min(sentences_len)))
    len_list = [a, b, c]
    name_list = ['0-256', '257-512', '>512']
    # plt.bar(range(len(len_list)), len_list, color='rgb', tick_label=name_list)
    # plt.show()


plot_len(all_sentences_train_len)
plot_len(all_sentences_test_len)
