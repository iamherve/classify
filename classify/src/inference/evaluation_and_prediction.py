import torch
from sklearn.metrics import f1_score
from classify.src.preprocessing.preprocessing import preprocess_text


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(test_loader)
    accuracy = sum(1 for p, l in zip(all_predictions, all_labels) if p == l) / len(
        all_labels
    )
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    return avg_loss, accuracy, f1


def prediction(model, text, label_to_idx, threshold=0.5):
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
