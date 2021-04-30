#!/usr/bin/env bash
#tensorflow_model_server --rest_api_port=9001 --model_config_file=models.config
#sudo docker run -td --name tfs_ner \
#          -p 8500:8500 \
#          --mount type=bind,source=/mnt/stephen-lib/stephen的个人文件夹/my_code/NLP算法组件研发/序列标注/NerAdapter/exported_model,target=/models/ner_model \
#          -t tensorflow/serving \
#          -e MODEL_NAME=ner-model --model_base_path=/models/ner-model/
sudo docker run --name tfs_ner \
          -p 8500:8500 \
          --mount type=bind,source=/mnt/stephen-lib/stephen的个人文件夹/my_code/NLP算法组件研发/序列标注/NerAdapter/bert_ce/exported_model,target=/models/ner_model \
          -e MODEL_NAME=ner_model -t tensorflow/serving