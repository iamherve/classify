import torch
import torch.nn as nn
import spacy
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from classify.data.raw import data
from classify.src.config.config import hyperparams

nlp = spacy.load("en_core_web_md")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    return tokens


def encode_labels(labels):
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_idx[label] for label in labels]
    return encoded_labels, label_to_idx


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(
        [torch.from_numpy(text) for text in texts], batch_first=True, padding_value=0
    )
    return texts_padded, torch.stack(labels)


texts, labels = zip(*data)
texts_preprocessed = [preprocess_text(text) for text in texts]
labels_encoded, label_to_idx = encode_labels(labels)


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


batch_size = hyperparams["batch_size"]
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    evaluation_dataset, batch_size=batch_size, collate_fn=collate_fn
)
