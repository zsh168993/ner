import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self,features,**kwargs):

        self.nums = len(features)
        self.input_ids = [torch.tensor(example.input_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels) for example in features]


    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'input_ids': self.input_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data



