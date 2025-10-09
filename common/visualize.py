import matplotlib.pyplot as plt


def visualize_loss(train_losses, val_losses, title, path):
    plt.plot(train_losses, color="red", label="train")
    plt.plot(val_losses, color="blue", label="val")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()