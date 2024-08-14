import torch
import random
import numpy as np
from classify.src.preprocessing.preprocessing import (
    preprocess_text,
    encode_labels,
)
from classify.src.training.training import train_and_evaluate
from classify.src.inference.evaluation_and_prediction import prediction
from classify.data.raw import data
from classify.data.inference import inference_phrases

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def main():
    # Preprocess the data
    texts, labels = zip(*data)
    texts_preprocessed = [preprocess_text(text) for text in texts]
    labels_encoded, label_to_idx = encode_labels(labels)

    # Train and evaluate the model
    model = train_and_evaluate(texts_preprocessed, labels_encoded, label_to_idx)

    # Perform prediction on a new article
    new_article = inference_phrases["entertainment"]
    predicted_categories, all_probabilities = prediction(
        model, new_article, label_to_idx
    )

    print("::::::::::::::::::::::::::::")
    print("New article: ", new_article)
    print("Predicted categories: ", predicted_categories)
    print("All probabilities:")
    for label, prob in all_probabilities:
        print(f"{label}: {prob:.4f}")
    print("::::::::::::::::::::::::::::")


if __name__ == "__main__":
    main()
