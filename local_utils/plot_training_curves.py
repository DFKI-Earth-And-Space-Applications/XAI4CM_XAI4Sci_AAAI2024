

import matplotlib.pyplot as plt

def plot_training_curves(df,save_dir,show=False):

    # Plot accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['train_accuracy'], label='Train Accuracy', marker='o')
    plt.plot(df['Epoch'], df['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir/'plot_accuracy_curves.png')
    plt.show()

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['Epoch'], df['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir/'plot_loss_curves.png')
    plt.show()

