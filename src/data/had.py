import os
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.base_dataset import BaseDataset
from src.data.RawBoost import process_Rawboost_feat


class HADDataModule(LightningDataModule):
    def __init__(
            self,
            root=None,
            test='dev',
            sample_rate=16000,
            resolution_train=0.02,
            resolution_test=0.02,
            max_label_len=200,
            add_label=False,
            batch_size=32,
            num_workers=4,
            # Data augmentation parameters
            use_augmentation=False,
            augmentation_algo=4,
            augmentation_prob=0.5,
    ) -> None:
        super().__init__()
        self.root = root
        self.test = test
        self.sample_rate = sample_rate
        self.resolution_train = resolution_train
        self.resolution_test = resolution_test
        self.max_label_len = max_label_len
        self.add_label = add_label
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Data augmentation parameters
        self.use_augmentation = use_augmentation
        self.augmentation_algo = augmentation_algo
        self.augmentation_prob = augmentation_prob

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_dataset(
                split='train',
                resolution=self.resolution_train,
                pad_mode='train'
            )
            self.valid_dataset = self.get_dataset(
                split='dev',
                resolution=self.resolution_test,
                pad_mode='test'
            )

        if stage == 'test' or stage is None:
            self.test_dataset = self.get_dataset(
                split=self.test,
                resolution=self.resolution_test,
                pad_mode='test'
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.valid_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn)

    def get_dataset(self, split, resolution, pad_mode):
        dataset = HADDataset(
            root=os.path.join(self.root, split),
            split=split,
            sample_rate=self.sample_rate,
            resolution=resolution,
            input_query='*.wav',
            label_root=os.path.join(self.root, 'segment_labels'),
            max_label_len=self.max_label_len,
            pad_mode=pad_mode,
            add_label=self.add_label,
            # Data augmentation parameters
            use_augmentation=self.use_augmentation,
            augmentation_algo=self.augmentation_algo,
            augmentation_prob=self.augmentation_prob,
        )
        return dataset


class HADDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        # Extract data augmentation parameters
        self.use_augmentation = kwargs.pop('use_augmentation', False)
        self.augmentation_algo = kwargs.pop('augmentation_algo', 4)
        self.augmentation_prob = kwargs.pop('augmentation_prob', 0.5)
        
        super().__init__(*args, **kwargs)

    def input_load_fn(self, path):
        audio, sr = sf.read(path)
        
        # Apply data augmentation if enabled and probability check passes
        if self.use_augmentation and random.random() < self.augmentation_prob:
            try:
                # Convert to numpy array if it's not already
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.numpy()
                else:
                    audio_np = audio
                
                # Apply RawBoost augmentation
                audio_aug = process_Rawboost_feat(audio_np, sr, self.augmentation_algo)
                
                # Convert back to original format
                if isinstance(audio, torch.Tensor):
                    audio = torch.from_numpy(audio_aug).float()
                else:
                    audio = audio_aug
                    
            except Exception as e:
                # If augmentation fails, keep original audio
                print(f"Data augmentation failed for {path}: {e}")
                pass
        
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
    HAD_dev_fake_00000003 0.00-1.26-T/1.26-2.12-F/2.12-3.04-T 0
    """
    resolutions = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    root = '/gpfs/sjtu/audiocc/data/import/deepfake/datasets/HAD/'
    parts = ['train', 'dev', 'test']
    out_dir = root + 'segment_labels'
    os.makedirs(out_dir, exist_ok=True)
    for part in parts:
        txt = root + f'label/HAD_{part}_label.txt'
        labels = [dict() for _ in resolutions]
        with open(txt, 'r') as f:
            for line in f:
                segs = line.strip().split(' ')
                id = segs[0]
                all_segs = segs[1].split('/')
                duration = float(all_segs[-1].split('-')[1])
                for i, res in enumerate(resolutions):
                    label = np.ones(int(duration / res))
                    for seg in all_segs:
                        times = seg.split('-')
                        start = float(times[0])
                        end = float(times[1])
                        if times[2] == 'F':
                            start_idx = int(start / res)
                            end_idx = int(end / res)
                            label[start_idx:end_idx] = 0
                    labels[i][id] = label
        for i, res in enumerate(resolutions):
            print(f"Saving labels for resolution {res}...")
            print(f"Total labels: {len(labels[i])}")
            np.save(os.path.join(out_dir, f'{part}_seglab_{res}.npy'),
                    labels[i], allow_pickle=True)


if __name__ == '__main__':
    # process_label()
    print("Loading HADDataModule...")
    data_module = HADDataModule(
        root='/gpfs/sjtu/audiocc/data/import/deepfake/datasets/HAD',
        test='dev',
        sample_rate=16000,
        resolution_train=0.02,
        max_label_len=200,
        add_label=False,
        batch_size=32,
        num_workers=4,
        # Enable data augmentation
        use_augmentation=True,
        augmentation_algo=4,
        augmentation_prob=0.5
    )
    data_module.setup()
    print("DataModule loaded successfully.")
    for batch in data_module.test_dataloader():
        utt_ids, inputs, labels, label_lens = batch
        for i in range(len(utt_ids)):
            print(f"Input: {inputs[i].shape}, Label: {labels[i].shape}, ")
            print(f"Utt ID: {utt_ids[i]}, Label: {labels[i]}")
        break