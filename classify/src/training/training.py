import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from classify.src.config.config import hyperparams
from classify.src.models.model import TextClassifier
from classify.src.preprocessing.preprocessing import (
    TextClassificationDataset,
    collate_fn,
)
from classify.src.inference.evaluation_and_prediction import evaluate_model
from classify.src.utils.plots import plot_loss_curves


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


def train_and_evaluate(texts_preprocessed, labels_encoded, label_to_idx):
    full_dataset = TextClassificationDataset(texts_preprocessed, labels_encoded)

    train_size = int(0.8 * len(full_dataset))
    evaluation_size = len(full_dataset) - train_size
    train_dataset, evaluation_dataset = random_split(
        full_dataset, [train_size, evaluation_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        evaluation_dataset, batch_size=hyperparams["batch_size"], collate_fn=collate_fn
    )

    output_dim = len(label_to_idx)

    model = TextClassifier(
        hyperparams["input_dim"],
        hyperparams["hidden_dim"],
        output_dim,
        dropout_rate=hyperparams["dropout_rate"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=hyperparams["weight_decay"])
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    best_model_state = None
    best_accuracy = 0

    train_losses = []
    test_losses = []

    for epoch in range(hyperparams["num_epochs"]):
        avg_loss = train_model(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy, test_f1 = evaluate_model(
            model, test_loader, criterion
        )

        train_losses.append(avg_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch+1}/{hyperparams['num_epochs']}",
            f"Train Loss: {avg_loss:.4f}",
            f"Test Loss: {test_loss:.4f}",
            f"Test Accuracy: {test_accuracy:.4f}",
            f"Test F1 Score: {test_f1:.4f}",
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()

        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with accuracy: {best_accuracy:.4f}")

    final_loss, final_accuracy, final_f1 = evaluate_model(model, test_loader, criterion)
    print(
        f"Final Test Loss: {final_loss:.4f}, Final Accuracy: {final_accuracy:.4f}, Final F1 Score: {final_f1:.4f}"
    )

    # Save the model
    torch.save(model.state_dict(), "text_classifier_model.pt")
    torch.save(label_to_idx, "label_to_idx.pt")
    print("Model and label mapping saved!")

    # Visualization
    plot_loss_curves(train_losses, test_losses)

    return model
