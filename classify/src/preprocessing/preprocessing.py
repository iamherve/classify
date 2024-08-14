import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from classify.data.raw import data
from classify.src.config.config import hyperparams
from classify.src.models.model import TextClassifier
from classify.data.inference import inference_phrases

nlp = spacy.load("en_core_web_md")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


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

output_dim = len(label_to_idx)

model = TextClassifier(
    hyperparams["input_dim"],
    hyperparams["hidden_dim"],
    output_dim,
    dropout_rate=hyperparams["dropout_rate"],
)

optimizer = optim.Adam(model.parameters(), weight_decay=hyperparams["weight_decay"])


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
early_stopping = EarlyStopping(patience=3, min_delta=0.01)

# Training loop with early stopping
best_model_state = None
best_accuracy = 0

for epoch in range(hyperparams["num_epochs"]):
    avg_loss = train_model(model, train_loader, criterion, optimizer)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(
        f"Epoch {epoch+1}/{hyperparams['num_epochs']}",
        f"Train Loss: {avg_loss:.4f}",
        f"Test Loss: {test_loss:.4f}",
        f"Test Accuracy: {test_accuracy:.4f}",
    )

    # Save the best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_state = model.state_dict().copy()

    # Early stopping check
    early_stopping(test_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break


# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with accuracy: {best_accuracy:.4f}")


final_loss, final_accuracy = evaluate_model(model, test_loader, criterion)
print(f"Final Test Loss: {final_loss:.4f}, Final Accuracy: {final_accuracy:.4f}")


def inference(model, text, threshold=0.5):
    model.eval()
    preprocessed_text = preprocess_text(text)
    input_tensor = torch.tensor(preprocessed_text).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output).squeeze()

    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    predicted_categories = []
    category_probabilities = []
    for idx, prob in enumerate(probabilities):
        if prob > threshold:
            label = idx_to_label[idx]
            predicted_categories.append(label)
            category_probabilities.append(prob)

    all_probabilities = [
        (idx_to_label[idx], prob.item()) for idx, prob in enumerate(probabilities)
    ]

    return predicted_categories, all_probabilities


new_article = inference_phrases["entertainment"]
predicted_categories, all_probabilities = inference(model, new_article)
print("::::::::::::::::::::::::::::")
print("New article: ", new_article)
print("Predicted categories: ", predicted_categories)
print("All probabilities:")
for label, prob in all_probabilities:
    print(f"{label}: {prob:.4f}")
print("::::::::::::::::::::::::::::")
# Save the model
torch.save(model.state_dict(), "text_classifier_model.pt")
torch.save(label_to_idx, "label_to_idx.pt")
print("Model and label mapping saved!!!!!")
