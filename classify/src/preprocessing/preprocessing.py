import torch
import torch.nn as nn
import spacy
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from classify.data.raw import data


nlp = spacy.load("en_core_web_md")


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

texts, labels = zip(*data)


def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    return tokens


texts_preprocessed = [preprocess_text(text) for text in texts]


def encode_labels(labels, all_labels):
    labels_to_index = {label: index for index, label in enumerate(all_labels)}

    multi_hot_encodings = torch.zeros(
        (len(labels), len(all_labels)), dtype=torch.float32
    )

    for i, label in enumerate(labels):
        label_index = labels_to_index[label]
        multi_hot_encodings[i, label_index] = 1.0

    return multi_hot_encodings


all_labels = sorted(set(labels))
labels_encoded = encode_labels(labels, all_labels)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(
        [torch.from_numpy(text) for text in texts], batch_first=True, padding_value=0
    )
    return texts_padded, torch.stack(labels)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


full_dataset = TextClassificationDataset(texts_preprocessed, labels_encoded)


train_size = int(0.8 * len(full_dataset))
evaluation_size = len(full_dataset) - train_size
train_dataset, evaluation_dataset = random_split(
    full_dataset, [train_size, evaluation_size]
)


batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    evaluation_dataset, batch_size=batch_size, collate_fn=collate_fn
)
