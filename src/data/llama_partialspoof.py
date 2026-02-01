import os
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.base_dataset import BaseDataset


class LlamaPartialSpoofDataModule(LightningDataModule):
    def __init__(
            self,
            root=None,
            part='cf',
            sample_rate=16000,
            resolution_train=0.02,
            resolution_test=0.02,
            max_label_len=200,
            add_label=False,
            batch_size=32,
            num_workers=4,

    ) -> None:
        super().__init__()
        self.root = root
        self.part = part
        self.sample_rate = sample_rate
        self.resolution_train = resolution_train
        self.resolution_test = resolution_test
        self.max_label_len = max_label_len
        self.add_label = add_label
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage='test'):
        assert stage == 'test', "Only 'test' stage is supported."
        self.test_dataset = self.get_dataset(
            split=self.part,
            resolution=self.resolution_test,
            pad_mode='test'
        )

    def train_dataloader(self):
        return DataLoader([], batch_size=1)

    def val_dataloader(self):
        return DataLoader([], batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn)

    def get_dataset(self, split, resolution, pad_mode):
        dataset = LlamaPartialSpoofDataset(
            root=self.root,
            split=split,
            sample_rate=self.sample_rate,
            resolution=resolution,
            input_query='*.wav',
            label_root=os.path.join(self.root, 'segment_labels'),
            max_label_len=self.max_label_len,
            pad_mode=pad_mode,
            add_label=self.add_label,
        )
        return dataset


class LlamaPartialSpoofDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_filter(self, sample_list):
        if self.split == 'cf':
            filtered_list = [f for f in sample_list if 'clean' in f or 'full' in f or 'cf' in f]
        elif self.split == 'cp':
            filtered_list = [f for f in sample_list if 'clean' in f or 'full' in f or 'cp' in f]
        elif self.split == 'oa':
            filtered_list = [f for f in sample_list if 'clean' in f or 'full' in f or 'oa' in f]
        else:
            raise Exception(f"Unrecognized split {self.split}.")
        print(f"Filtered {len(sample_list)} -> {len(filtered_list)} samples for part {self.split}.")
        return filtered_list

    def input_load_fn(self, path):
        audio, sr = sf.read(path)
        return audio

    def label_load_fn(self):
        if isinstance(self.resolution, list):
            labels = []
            for res in self.resolution:
                label_file = os.path.join(self.label_root,
                                          f'{self.split}_seglab_{res}.npy')
                label = np.load(label_file, allow_pickle=True).item()
                labels.append(label)
        else:
            label_file = os.path.join(self.label_root,
                                      f'{self.split}_seglab'
                                      f'_{self.resolution}.npy')
            labels = np.load(label_file, allow_pickle=True).item()
        return labels

    def pad(self, items):
        if self.pad_mode == 'train':
            utt_id, input, label, label_len = items[:4]

            if isinstance(label, list):  # multi-resolution labels
                scale = int(self.sample_rate * self.resolution[0])
                input_len = self.max_label_len[0] * scale
                if len(input) < input_len:
                    input = F.pad(input,
                                  (0, input_len - len(input)),
                                  mode='constant', value=0)

                if label_len[0] < self.max_label_len[0]:
                    for i, l in enumerate(label):
                        max_len_i = self.max_label_len[i]
                        label[i] = F.pad(l, (0, max_len_i - len(l)),
                                         mode='constant', value=0)
                    new_items = (utt_id, input, label, label_len)

                else:
                    start = random.randint(0,
                                           label_len[0] - self.max_label_len[0])
                    input = input[start * scale:(start + self.max_label_len[
                        0]) * scale]
                    if len(input) < input_len:
                        input = F.pad(input,
                                      (0, input_len - len(input)),
                                      mode='constant', value=0)
                    for i, l in enumerate(label):
                        start_i = start // (2 ** i)
                        max_len_i = self.max_label_len[i]
                        label[i] = l[start_i:start_i + max_len_i]
                    new_items = (utt_id, input, label, label_len)

            else:  # single resolution label
                scale = int(self.sample_rate * self.resolution)

                if len(input) < self.max_label_len * scale:
                    input = F.pad(input,
                                  (0, self.max_label_len * scale - len(input)),
                                  mode='constant', value=0)

                if label_len < self.max_label_len:
                    label = F.pad(label, (0, self.max_label_len - label_len),
                                  mode='constant', value=0)
                    new_items = (utt_id, input, label, label_len)

                    if self.add_label:
                        bound_label = items[4]
                        bound_len = items[5]
                        bound_label = F.pad(bound_label,
                                            (0,
                                             self.max_label_len - len(
                                                 bound_label)),
                                            mode='constant', value=0)
                        new_items += (bound_label, bound_len)

                else:
                    start = random.randint(0, label_len - self.max_label_len)
                    input = input[
                            start * scale:(start + self.max_label_len) * scale]
                    if len(input) < self.max_label_len * scale:
                        input = F.pad(input,
                                      (0,
                                       self.max_label_len * scale - len(input)),
                                      mode='constant', value=0)
                    label = label[start:start + self.max_label_len]
                    new_items = (utt_id, input, label, label_len)

                    if self.add_label:
                        bound_label = items[4]
                        bound_len = items[5]
                        bound_label = bound_label[
                                      start:start + self.max_label_len]
                        new_items += (bound_label, bound_len)

        else:  # pad_mode == 'test'
            utt_id, input, label, label_len = items[:4]
            scale = int(self.sample_rate * self.resolution)
            input = F.pad(input, (0, label_len * scale - len(input)),
                          mode='constant', value=0)
            new_items = (utt_id, input, label, label_len)
            if self.add_label:
                new_items += (items[4], items[5])

        return new_items

    def collate_fn(self, batch):
        utt_ids, inputs, labels, label_lens = zip(*batch)
        inputs = torch.stack(inputs)
        if isinstance(labels[0], list):
            # multi-resolution labels
            num_res = len(labels[0])
            grouped_labels = [[] for _ in range(num_res)]
            grouped_lens = [[] for _ in range(num_res)]
            for lbls, lens in zip(labels, label_lens):
                for i in range(num_res):
                    grouped_labels[i].append(lbls[i])
                    grouped_lens[i].append(lens[i])
            # Stack across batch
            labels = [torch.stack(g) for g in grouped_labels]
            label_lens = [torch.tensor(g) for g in grouped_lens]
        else:
            # single resolution labels
            labels = torch.stack(labels)
            label_lens = torch.tensor(label_lens)
        return list(utt_ids), inputs, labels, label_lens


def process_label():
    """
    Label format:
    dev01-cosyvoice-partial-cp_1272_128104_000005_000014 5.6150 spoof
    0.0000-0.3800-bonafide 0.3800-1.8100-spoof ...
    """
    resolutions = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    labels = [dict() for _ in resolutions]
    root = '/data/import/deepfake/LlamaPartialSpoof/'
    txts = ['label_R01TTS.0.a.txt', 'label_R01TTS.0.b.txt']
    out_dir = '/data/import/deepfake/LlamaPartialSpoof/segment_labels'
    for txt in txts:
        txt = root + txt
        with open(txt, 'r') as f:
            for line in f:
                segs = line.strip().split(' ')
                id = segs[0]
                duration = float(segs[1])
                all_segs = segs[3:]  #
                for i, res in enumerate(resolutions):
                    label = np.ones(int(duration / res))
                    for seg in all_segs:
                        times = seg.split('-')
                        start = float(times[0])
                        end = float(times[1])
                        if times[2] == 'spoof':
                            start_idx = int(start / res)
                            end_idx = int(end / res)
                            label[start_idx:end_idx] = 0
                    labels[i][id] = label
    for i, res in enumerate(resolutions):
        print(f"Saving labels for resolution {res}...")
        print(f"Total labels: {len(labels[i])}")
        np.save(os.path.join(out_dir, f'seglab_{res}.npy'), labels[i],
                allow_pickle=True)

    # select different parts: cf, cp, oa
    for npy in os.listdir(out_dir):
        file = os.path.join(out_dir, npy)
        labels = np.load(file, allow_pickle=True).item()
        labels_part = {'real': dict(), 'full': dict(),
                       'cf': dict(), 'cp': dict(), 'oa': dict()}
        for k, v in labels.items():
            if 'clean' in k:  # real
                labels_part['real'][k] = v
            elif 'full' in k:
                labels_part['full'][k] = v
            elif 'cf' in k:
                labels_part['cf'][k] = v
            elif 'cp' in k:
                labels_part['cp'][k] = v
            elif 'oa' in k:
                labels_part['oa'][k] = v
            else:
                raise Exception(f"Unrecognized key {k} in {npy}.")

        for part in ['cf', 'cp', 'oa']:
            labels_new = dict()
            labels_new.update(labels_part['real'])
            labels_new.update(labels_part['full'])
            labels_new.update(labels_part[part])
            out_file = os.path.join(out_dir, f"{part}_{npy}")
            print(
                f"Saving {part} labels to {out_file}, total {len(labels_new)} samples.")
            np.save(out_file, labels_new, allow_pickle=True)


if __name__ == '__main__':
    # process_label()
    print("Loading LlamaPartialSpoofDataModule...")
    data_module = LlamaPartialSpoofDataModule(
        root='/data/import/deepfake/LlamaPartialSpoof',
        part='cf',
        sample_rate=16000,
        resolution_train=0.02,
        max_label_len=200,
        # resolution_train=[0.02, 0.04, 0.08, 0.16],
        # max_label_len=[200, 100, 50, 25],
        add_label=False,
        batch_size=32,
        num_workers=4
    )
    data_module.setup()
    print("DataModule loaded successfully.")
    for batch in data_module.test_dataloader():
        utt_ids, inputs, labels, label_lens = batch
        for i in range(len(utt_ids)):
            print(f"Input: {inputs[i].shape}, Label: {labels[i].shape}, ")
            print(f"Utt ID: {utt_ids[i]}, Label: {labels[i]}")
        break