# NerAdapter

本项目针对NLP的NER方向开源目前业界流行的解决方案，该方案提供了从线下训练到线上部署的一整套闭环流程。<br>
具体模型包括：<br>
1）基于bert进行微调【已集成】；<br>
2）bert+crf【已集成】；<br>
3）bert+bilstm+crf【已集成】；<br>
4）bert+mrc机制；<br>

[知乎入口]()

[github入口](https://github.com/Vincent131499/NerAdapter)

## 1.项目构成

**【整体环境配置】**

- Python3.6
- tensorflow-gpu==1.14.0
- tensorflow-serving-api==1.14.0

为了增加独立性，方便复现，将每一个模型作为单独的模块进行维护。项目构成如下所示：

- ---bert_ce：基于bert微调的方案
  - ---run_ner_bert_ce.py：ner训练的主程序入口；
  - ---modeling.py：引自bert源码；
  - ---optimization.py：引自bert源码；
  - ---tokenization.py：引自bert源码；
  - ---tf_metrics.py：性能指标计算；
  - ---utils_.py：工具函数封装；
  - ---infer_offline.py：线下推理；
  - ---infer_online.py：线上推理；
  - ---checkpoint：存储模型的文件夹；
  - ---exported_model：存储导出的pb格式模型的文件夹；
  - ---start_tfs_single_model.sh：启动docker的tf-serving容易的命令；

- ---bert_blstm_crf：bert+crf和bert+bilstm+crf两种方案
  - ---run_ner_bert_blstm_crf.py：ner训练的主程序入口；
  - ---modeling.py：引自bert源码；
  - ---optimization.py：引自bert源码；
  - ---tokenization.py：引自bert源码；
  - ---tf_metrics.py：性能指标计算；
  - ---utils_.py：工具函数封装；
  - ---infer_offline.py：线下推理；
  - ---infer_online.py：线上推理；
  - ---checkpoint：存储模型的文件夹；
  - ---exported_model：存储导出的pb格式模型的文件夹；
  - ---start_tfs_single_model.sh：启动docker的tf-serving容易的命令；

- ---china-people-daily-data

  中国人民日报的NER数据集，包括train/dev/test，格式如下：

  ```bash
  海 钓 比 赛 地 点 在 厦 门 与 金 门 之 间 的 海 域 。-seq-O O O O O O O B-LOC I-LOC O B-LOC I-LOC O O O O O O
  ```

- ---china-people-daily-data-lie

  同是中国人民日报的NER数据集，只不过格式发生了变化：

  ```bash
  海 O
  钓 O
  比 O
  赛 O
  地 O
  点 O
  在 O
  厦 B-LOC
  门 I-LOC
  与 O
  金 B-LOC
  门 I-LOC
  之 O
  间 O
  的 O
  海 O
  域 O
  。 O
  ```

- ---msra_data

  MSRA的NER数据集，包含内容以及数据格式与‘china-people-daily-data’一致。

- ---msra_data_lie

  MSRA的NER数据集，包含内容以及数据格式与‘china-people-daily-data-lie’一致。

## 2.项目运行

接下来将针对**bert_ce**和**bert_bilstm_crf**这两个模型来叙述从线下训练到线上部署的闭环流程。

**【前置：预训练模型配置】**

下载chinese_L-12_H-768_A-12(链接：https://pan.baidu.com/s/1uLrGC6jDWZkf85QdaTLHwQ 
提取码：ll38)，并将放置到‘pretrained_models’文件夹中；

### 2.1 bert_ce

该模型直接在BERT的后面加入一个全连接层，然后使用交叉熵损失函数进行学习优化指标，具体模型图如下所示：

![bert_ce模型图](https://z3.ax1x.com/2021/04/30/gAapVO.jpg)

#### 2.1.1 数据准备

准备训练的数据集，包含train/dev/test.txt，格式如下：

```bash
海 O
钓 O
比 O
赛 O
地 O
点 O
在 O
厦 B-LOC
门 I-LOC
与 O
金 B-LOC
门 I-LOC
之 O
间 O
的 O
海 O
域 O
。 O
```

本项目直接使用china-people-daily-data-lie数据集，包含‘PER’(人物)、‘ORG’(组织机构)、‘LOC’(地点)三种实体类型；

#### 2.1.2 模型训练

进入bert_ce文件夹，运行‘run_ner_bert.py’；

注意设置下列参数：

```bash
do_train：True
do_eval：True
do_predict：False
do_infer：False
```

```bash
$ cd bert_ce
$ python run_ner_bert.py
```

运行后取得的评估结果如下：

![bert_ce评估结果](https://z3.ax1x.com/2021/04/30/gAakRA.png)

关于模型的其它参数已封装在run_ner_bert.py文件中，可自行查看。

#### 2.1.3 模型导出

针对‘run_ner_bert.py’设置如下参数：

```python
do_train：False
do_eval：False
do_predict：True
do_infer：True
```

随后再次运行：

```bash
$ python run_ner_bert.py
```

运行后可在exported_model文件中查看导出的pb模型：

![导出结果](https://z3.ax1x.com/2021/04/30/gAaAxI.png)

#### 2.1.4 线下推理

运行‘infer_offline.py’：

```bash
$ python infer_offline.py
```

![线下demo](https://z3.ax1x.com/2021/04/30/gAaiPH.png)

#### 2.1.5 线上服务

线上服务是基于docker和tf-serving来启动的。

首先安装docker，然后拉取tf-serving镜像：

```bash
$ sudo docker pull tensorflow/serving	
```

随后使用脚本来启动服务：

```bash
$ bash start_tfs_single_model.sh
```

![tf-serving启动](https://z3.ax1x.com/2021/04/30/gAaVMt.png)

启动成功后运行‘infer_online.py’：

```bash
$ python infer_online.py
```

![线上demo](https://z3.ax1x.com/2021/04/30/gAaFGd.png)

### 2.2 bert_bilstm_crf

该模型是在BERT之后加入一个双向LSTM网络，随后使用CRF进行解码优化，具体模型图如下：

![bert_bilstm_crf模型图](https://z3.ax1x.com/2021/04/30/gAa9aD.jpg)

后面的训练到部署事宜与2.1节一致，请自行查看。

在使用这个方案时，若你想切换bert_bilstm_crf和bert_crf两种模型，可在‘bert_blstm_crf/models.py’文件中设置：

```python
#位于models.py文件的第101行
#crf_only=True：表示模型为bert_crf；
#crf_only=False：表示模型为bert_bilstm_crf；
rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
```

## 3.后续计划

后续将继续集成NER相关的模型，一种是传统NER性能优化，一种是嵌套问题解决；

模型如下：

- [ ] [MRC-MER](https://arxiv.org/pdf/1910.11476.pdf)
- [ ] [FLAT-NER](https://www.aclweb.org/anthology/2020.acl-main.611.pdf)


**最后祝大家5.1快乐！**

#### **[参考]**

[1] [BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)

[2] [BERT-NER](https://github.com/kyzhouhzau/BERT-NER)
