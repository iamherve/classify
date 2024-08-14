import spacy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

nlp = spacy.load("en_core_web_md")


def preprocess_text(text):
    doc = nlp(text)
    return [
        token.vector
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]


def encode_labels(labels):
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_idx[label] for label in labels]
    return encoded_labels, label_to_idx


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(
        [torch.tensor(text) for text in texts], batch_first=True
    )
    return texts_padded, torch.tensor(labels)


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
