import matplotlib.pyplot as plt


def plot_loss_curves(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o"
    )
    plt.plot(
        range(1, len(test_losses) + 1), test_losses, label="Evaluation Loss", marker="o"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()
