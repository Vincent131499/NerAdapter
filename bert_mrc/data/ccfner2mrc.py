# encoding: utf-8

import os
import json


def convert_file(input_file, output_file, tag2query_file):
    """
    Convert MSRA raw data to MRC format
    """
    origin_count = 0
    new_count = 0
    tag2query = json.load(open(tag2query_file, encoding='utf-8'))
    mrc_samples = []
    index1 = 0
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = json.loads(line.strip())
            text = line['text']
            tags_entities = line.get('label', None)
            words = list(text)
            if not line:
                continue
            origin_count += 1
            index2 = 0
            for label, query in tag2query.items():
                assert len(query) == 9

                start_position = []
                end_position = []
                if tags_entities is not None:
                    for key, value in tags_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name  # 处理的文本索引不能超
                                if key == label:
                                    start_position.append(start_index)
                                    end_position.append(end_index)

                mrc_samples.append(
                    {
                        "context": text,
                        "start_position": start_position,
                        "end_position": end_position,
                        "query": query,
                        "qas_id": str(index1) + '.' + str(index2),
                        "entity_label": label
                    }
                )
                index2 += 1
                new_count += 1
            index1 += 1

    json.dump(mrc_samples, open(output_file, "w", encoding='utf-8'), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")


def main():
    msra_raw_dir = "ccfner"
    msra_mrc_dir = "ccfner_mrc_format"
    tag2query_file = "queries/ccf_ner.json"
    os.makedirs(msra_mrc_dir, exist_ok=True)
    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(msra_raw_dir, f"{phase}.json")
        new_file = os.path.join(msra_mrc_dir, f"mrc-ner.{phase}")
        convert_file(old_file, new_file, tag2query_file)


if __name__ == '__main__':
    main()
