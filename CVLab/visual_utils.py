import matplotlib.pyplot as plt

def save_losses(training_losses: list[float], eval_losses: list[float], path: str) -> None:
    """
    Takes in training losses and evaluation losses and saves plots to path-directory.
    """
    plt.clf() # Clear figure
    epochs = len(training_losses)
    plt.plot(range(epochs), training_losses, label='Train loss')
    plt.plot(range(epochs), eval_losses, label='Evaluation loss')
    plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.grid(); plt.legend(); plt.savefig(path + ".png", dpi=300)