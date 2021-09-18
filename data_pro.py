"""Read, split and save the kaggle dataset for our model"""
import random
import tqdm
import os
import json
from transformers import BertTokenizer
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.pseudo = pseudo
        self.distant_labels = distant_labels


class BaseFeature:
    def __init__(self,
                 input_ids,
                 attention_masks,
                 token_type_ids,
                 labels):
        # BERT 输入
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels
class InputExample:
    def __init__(self,
                 text,
                 labels=None,
                ):

        self.text = text
        self.labels = labels

def get_examples(raw_examples_path):
    examples = []
    with open(raw_examples_path,"r",encoding="utf-8") as f:
        data_example = json.load(f)
    for i, item in enumerate(data_example):
        examples.append(InputExample(
                                     text=item["text"],
                                     labels=item["tag"],
                                    ))
    return examples
ENTITY_TYPES = {"O":0, 'B-PER':1, 'B-LOC':2, 'B-ORG':3, 'B-GPE':4,
                'M-PER':5, 'M-LOC':6, 'M-ORG':7, 'M-GPE':8,
                'E-PER':9, 'E-LOC':10, 'E-ORG':11, 'E-GPE':12,
                'S-PER':13, 'S-LOC':14, 'S-ORG':15, 'S-GPE':16}
def convert_examples_to_features(examples, max_seq_len=101, bert_dir="bert_model",):
    tokenizer =BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []

    for i, example in enumerate(examples):
        encode_dict = tokenizer.encode_plus(text=example.text,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)
        label_ids=[]
        for i in example.labels.split():
            label_ids.append(ENTITY_TYPES[i])
        label_ids = [0] + label_ids + [0]

        # pad
        if len(label_ids) < max_seq_len:
            pad_length = max_seq_len - len(label_ids)
            label_ids = label_ids + [0] * pad_length  #
        input_ids = encode_dict['input_ids']
        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']

        if len(input_ids)!=len(label_ids):
            print(123)
        feature = BaseFeature(
            # bert inputs
            input_ids=input_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids,
            labels=label_ids,

        )
        features.append(feature)
    return features
if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/char_ner_train.csv'
    train_path="data/train.json"
    dev_path = "data/dev.json"
    text=[]
    label=[]
    textlen=[]
    data_list=[]

    # Load the dataset into memory
    with open(path_dataset, encoding='utf-8') as fp,\
            open(train_path, 'w',encoding='utf-8')as f_train\
                    ,open(dev_path, 'w', encoding='utf-8') as f_dev:
        for line in tqdm.tqdm(fp, desc='Tokenizing'):
            temp = line.strip()
            temp = temp.split(",")
            if temp[1]=="tag":
                continue
            if temp[1] =="":
                textlen.append(len(text))
                text=" ".join(text)
                label = " ".join(label)
                data={'text': text, 'tag': label}
                data_list.append(data)
                text = []
                label = []
            elif len(text)>98:
                textlen.append(len(text))
                text = " ".join(text)
                label = " ".join(label)
                data = {'text': text, 'tag': label}
                data_list.append(data)
                text = []
                label = []
                text.append(temp[0])
                label.append(temp[1])
            else:
                text.append(temp[0])
                label.append(temp[1])
        random.shuffle(data_list)
        json.dump(data_list[14000:14002], f_dev, ensure_ascii=False, indent=2)
        json.dump(data_list[:14000], f_train, ensure_ascii=False, indent=2)
    print("长度：",textlen)
    print("句子数量：", len(textlen))
    print("最大句子：", max(textlen))
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.rcParams["font.sans-serif"] = ["SimHei"]

    mpl.rcParams["axes.unicode_minus"] = False
    # 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
    def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
        plt.hist(myList, 100)
        plt.xlabel(Xlabel)
        plt.xlim(Xmin, Xmax)
        plt.ylabel(Ylabel)
        plt.ylim(Ymin, Ymax)
        plt.title(Title)
        plt.show()


    draw_hist(textlen, '句子分布图', '长度区间', '样本数', 4, 100, 0.0, 600)  # 直方图展示

    print("- done.")