from abc import ABC
import torch
import os
import copy
import json


class DataProcessor(object):
    @classmethod
    def _read_csv(cls, input_file, delimiter="\t"):
        return []

    @classmethod
    def _read_text(cls, input_file):
        dlines = []
        with open(input_file, 'r', encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                dlines.append(line)
        return dlines

    @classmethod
    def _read_json(cls, input_file):
        return []


class InputExample(object):
    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, input_len, label_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_idx = label_idx
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    all_input_ids, all_attention_mask, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    return all_input_ids, all_attention_mask, all_labels, all_lens


def convert_examples_to_features(examples, max_seq_length, label2id, pad_token=0,
                                 vocab_dict=None):
    features = []
    for (ex_index, example) in enumerate(examples):
        word_list = example.text_a.split()
        input_ids = [vocab_dict.get(word, 1) for word in word_list if word.strip()]
        label_id = label2id[example.label]
        input_mask = [1] * len(input_ids)
        input_len = sum(input_mask)
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
            input_mask = input_mask[:max_seq_length]
        else:
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        if ex_index < 5:
            print("*** Example ***")
            print(f"guid: {example.guid}")
            print(f"tokens: {' '.join([str(x) for x in word_list])}")
            print(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            print(f"input_mask: {' '.join([str(x) for x in input_mask])}")
            print(f"label_id: {example.label}")
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      input_len=input_len, label_idx=label_id))
    return features


class WordsProcessor(DataProcessor, ABC):
    word_type = None

    def __init__(self, data_dir, word_type=True, ch_flag=True):
        self.ch_flag = False
        self.word_type = word_type
        self.vocab_dict = {"UNKNOWN": 1, "PADDING": 0}
        tmp_label_set = set()
        with open(f"{data_dir}/train.txt", "r", encoding="utf-8") as fr:
            for line in fr:
                json_data = json.loads(line.strip())
                word_list = json_data['words'].split()
                label = json_data['label']
                tmp_label_set.add(label)
                for i, word in enumerate(word_list):
                    if word not in self.vocab_dict:
                        self.vocab_dict[word] = len(self.vocab_dict)
        label_file = f"{data_dir}/labels.txt"
        if os.path.exists(label_file):
            self.label_list = []
            with open(label_file, 'r', encoding="utf-8") as fr:
                for line in fr:
                    self.label_list.append(line.strip())
        else:
            print("label.txt does not exist")
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return self.label_list, self.label2id, self.id2label

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            json_data = json.loads(line.strip())
            words = json_data['words']
            label = json_data['label']
            guid = "%s-%s" % (set_type, i)
            text = words
            if self.word_type:
                examples.append(InputExample(guid=guid, text_a=words, label=label))
            else:
                examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples
