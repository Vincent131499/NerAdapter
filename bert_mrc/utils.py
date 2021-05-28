import random
import os
import numpy as np
import torch
import pickle
import json

from pathlib import Path

category = ["address", "book", "company", 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
# category = ['NT', 'NS', 'NR'] #组织结构、地理位置、人名
max_len = 128


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''

    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    data = []
    with open(str(file_path), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            data.append(dic)
    return data


def json_to_text(file_path, data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w', encoding='utf-8') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
