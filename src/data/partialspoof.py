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


class PartialSpoofDataModule(LightningDataModule):
    def __init__(
            self,
            root=None,
            sample_rate=16000,
            resolution_train=0.16,
            resolution_test=0.16,
            max_label_len=25,
            add_label=False,
            batch_size=32,
            num_workers=4,
            debug_mode=False,
            max_samples=None,
            # Data augmentation parameters
            use_augmentation=False,
            augmentation_algo=4,
            augmentation_prob=0.5,

    ) -> None:
        super().__init__()
        self.root = root
        self.sample_rate = sample_rate
        self.resolution_train = resolution_train
        self.resolution_test = resolution_test
        self.max_label_len = max_label_len
        self.add_label = add_label
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_mode = debug_mode
        self.max_samples = max_samples
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
                split='eval',
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
        dataset = PartialSpoofDataset(
            root=os.path.join(self.root, split),
            split=split,
            sample_rate=self.sample_rate,
            resolution=resolution,
            input_query='*.wav',
            label_root=os.path.join(self.root, 'segment_labels'),
            max_label_len=self.max_label_len,
            pad_mode=pad_mode,
            add_label=self.add_label,
            debug_mode=self.debug_mode,
            max_samples=self.max_samples,
            # Data augmentation parameters
            use_augmentation=self.use_augmentation,
            augmentation_algo=self.augmentation_algo,
            augmentation_prob=self.augmentation_prob,
        )
        return dataset


class PartialSpoofDataset(BaseDataset):
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
        label_file = os.path.join(self.label_root,
                                  f'{self.split}_seglab_{self.resolution}.npy')
        labels = np.load(label_file, allow_pickle=True).item()
        labels = {k: v.astype(int) for k, v in labels.items()}
        return labels

    def add_other_label(self, items):
        utt_id, input, ori_label, ori_label_length = items
        bound_labels_root = (
            f'{self.label_root}/bound_{self.resolution}_labels/{self.split}/{utt_id}_bound.npy')
        bound_label = np.load(bound_labels_root).astype(np.float32)
        bound_label = torch.FloatTensor(bound_label)
        bound_len = len(bound_label)
        new_items = (utt_id, input, ori_label, ori_label_length,
                     bound_label, bound_len)
        return new_items

    def pad(self, items):
        if self.pad_mode == 'train':
            utt_id, input, label, label_len = items[:4]
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
                                         self.max_label_len - len(bound_label)),
                                        mode='constant', value=0)
                    new_items += (bound_label, bound_len)

            else:
                start = random.randint(0, label_len - self.max_label_len)
                input = input[
                        start * scale:(start + self.max_label_len) * scale]
                if len(input) < self.max_label_len * scale:
                    input = F.pad(input,
                                  (0, self.max_label_len * scale - len(input)),
                                  mode='constant', value=0)
                label = label[start:start + self.max_label_len]
                new_items = (utt_id, input, label, label_len)
                if self.add_label:
                    bound_label = items[4]
                    bound_len = items[5]
                    bound_label = bound_label[start:start + self.max_label_len]
                    new_items += (bound_label, bound_len)
        else:
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
        labels = torch.stack(labels)
        label_lens = torch.tensor(label_lens)
        return list(utt_ids), inputs, labels, label_lens


if __name__ == '__main__':
    print("Loading PartialSpoofDataModule...")
    data_module = PartialSpoofDataModule(
        root='/data/import/deepfake/PartialSpoof',
        sample_rate=16000,
        resolution_train=0.16,
        max_label_len=25,
        add_label=False,
        batch_size=32,
        num_workers=4,
        # Enable data augmentation
        use_augmentation=True,
        augmentation_algo=4,  # Use algorithm 4 (1+2+3: LnL + ISD + SSI)
        augmentation_prob=0.5  # 50% probability of applying augmentation
    )
    data_module.setup()
    print("DataModule loaded successfully.")
    print("Data augmentation enabled with algorithm 4 and 50% probability.")
