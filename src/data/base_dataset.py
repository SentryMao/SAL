import os

import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.utils import find_files

class BaseDataset(Dataset):
    def __init__(
            self,
            root=None,
            split=None,
            sample_rate=None,
            resolution=None,
            max_input_len=None,
            min_input_len=None,
            input_query=None,
            label_root=None,
            max_label_len=None,
            pad_mode='label',
            add_label=False,
            debug_mode=False,
            max_samples=None,

    ) -> None:
        super().__init__()
        self.root = root
        self.input_query = input_query
        self.max_input_len = max_input_len
        self.min_input_len = min_input_len
        self.split = split
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.pad_mode = pad_mode
        self.add_label = add_label
        self.label_root = label_root
        self.max_label_len = max_label_len
        self.debug_mode = debug_mode
        self.max_samples = max_samples

        if isinstance(root, str):
            sample_list = sorted(find_files(root, query=self.input_query))
        elif isinstance(root, list):
            sample_list = []
            for r in root:
                sample_list += sorted(find_files(r, query=self.input_query))
        else:
            raise AttributeError(f'{root} root is not a list or str, {root}')

        # filter by min length
        if min_input_len is not None:
            sample_length = [self.input_load_fn(f).shape[-1] for f in
                             sample_list]
            idxs = [idx for idx in range(len(sample_list)) if
                    sample_length[idx] > min_input_len]
            if len(sample_list) != len(idxs):
                print(
                    "some files are filtered by audio length threshold "
                    f"({len(sample_list)} -> {len(idxs)})."
                )
            sample_list = [sample_list[idx] for idx in idxs]

        # assert the number of files
        assert len(sample_list) != 0, f"Not found any sample files in ${root}."

        # filter audio sample or segment
        self.labels = self.label_load_fn()
        self.sample_list = self.sample_filter(
            sample_list)
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in
                        self.sample_list]
        print(f"Found {len(self.sample_list)} samples in {self.split} set.")
        
        # Debug模式：限制数据量
        if self.debug_mode and self.max_samples is not None:
            if len(self.sample_list) > self.max_samples:
                print(f"Debug mode: limiting samples from {len(self.sample_list)} to {self.max_samples}")
                self.sample_list = self.sample_list[:self.max_samples]
                self.utt_ids = self.utt_ids[:self.max_samples]

    def sample_filter(self, sample_list):
        return sample_list

    def input_load_fn(self, path):
        raise NotImplementedError

    def label_load_fn(self):
        raise NotImplementedError

    def add_other_label(self, items):
        raise NotImplementedError

    def pad(self, items):
        raise NotImplementedError

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        input = self.input_load_fn(self.sample_list[index])
        input = torch.FloatTensor(input)

        label = self.labels[utt_id]
        label = torch.FloatTensor(label)

        ori_label_len = len(label)
        items = (utt_id, input, label, ori_label_len,)

        # add other supervise information
        if self.add_label:
            items = self.add_other_label(items)

        # pad audio and label to certain length
        if self.pad_mode is not None:
            items = self.pad(items)

        return items

    def __len__(self):
        return len(self.utt_ids)

    def get_length_list(self):
        length_list = []
        for f in self.sample_list:
            audio = self.input_load_fn(f)
            length_list.append(audio.shape[-1])
        return length_list
