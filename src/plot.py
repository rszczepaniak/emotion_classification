import json
import os
from datetime import timedelta

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def insert_data_into_json(dataset_name: str, plot_name: str, num_epochs: int, train_acc, eval_acc, train_time,
                          json_data_path: str = "models_training_results.json") -> None:
    with open(json_data_path, "r") as json_file:
        data = json.load(json_file)

    data.append({
        "dataset_name": dataset_name,
        "name": plot_name,
        "num_epochs": num_epochs,
        "train_accuracy_per_epoch": train_acc,
        "eval_accuracy_per_epoch": eval_acc,
        "train_time": train_time,
    })
    with open(json_data_path, "w") as json_file:
        json.dump(data, json_file)


def plot_all_models(dataset_name: str, num_epochs: int, name_prefixes: list[str], exact_names: list[str],
                    filters: list[str]) -> None:
    with open("models_training_results.json", "r") as json_file:
        data = json.load(json_file)
    colors = ("orange", "blue", "green", "red", "purple", "cyan", "yellow")
    color_id = 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plt.title(" ".join(name_prefixes))
    # Plot training accuracy
    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)  # Add horizontal gridlines

    # Plot evaluation accuracy
    axes[1].set_title('Evaluation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)  # Add horizontal gridlines

    if exact_names:
        fun = lambda model: model["dataset_name"] == dataset_name and model["num_epochs"] == num_epochs and model[
            "name"] in exact_names
    elif filters:
        fun = lambda model: model["dataset_name"] == dataset_name and model["num_epochs"] == num_epochs and any(
            map(lambda filter: filter in model["name"], filters))
    elif name_prefixes:
        fun = lambda model: model["dataset_name"] == dataset_name and model["num_epochs"] == num_epochs and any(
            map(lambda prefix: prefix in model["name"], name_prefixes))
    else:
        fun = lambda model: model["dataset_name"] == dataset_name and model["num_epochs"] == num_epochs

    for trained_model in data:
        if fun(trained_model):
            # Plot training accuracy
            axes[0].plot(
                range(1, num_epochs + 1),
                trained_model["train_accuracy_per_epoch"],
                label=trained_model["name"],
                color=colors[color_id]
            )
            # Plot evaluation accuracy
            axes[1].plot(
                range(1, num_epochs + 1),
                trained_model["eval_accuracy_per_epoch"],
                label=trained_model["name"],
                color=colors[color_id]
            )
            color_id += 1

        # Add legends
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_emotion_accuracies(dataset_name: str, num_epochs: int, name: str, num_emotions: int = 8,
                            plot_train: bool = True, save_plot: bool = False) -> None:
    """
    Plots train or eval accuracies for each emotion for the given dataset and model, averaged across all chunks.

    Args:
        dataset_name (str): Name of the dataset.
        num_epochs (int): Number of epochs to plot.
        name (str): Name of the model/plot.
        num_emotions (int): Number of emotions (default is 8).
        plot_train (bool): Whether to plot training accuracies (default is True). If False, plots evaluation accuracies.
        save_plot (bool): Whether to save the plot as an image.
        save_path (str): Directory to save the plot if save_plot is True.
    """
    # Check if file exists
    json_file = "models_training_results.json"
    if not os.path.exists(json_file):
        print(f"Error: {json_file} does not exist.")
        return

    # Load JSON data
    with open(json_file, 'r') as f:
        all_models = json.load(f)

    # Verify that the dataset and name match
    data = None
    for model in all_models:
        if model["dataset_name"] == dataset_name and model["name"] == name and model["num_epochs"] == num_epochs:
            data = model
            break
    if data is None:
        print(f"Error: Model with dataset '{dataset_name}' and name '{name}' not found in the file.")
        return

    # Extract training and evaluation accuracies per emotion
    train_accuracies = data["train_accuracy_per_epoch"]
    eval_accuracies = data["eval_accuracy_per_epoch"]

    def calculate_mean_accuracies(acc_dict):
        mean_accuracies = {epoch: [0] * num_emotions for epoch in range(num_epochs)}

        for chunk_id, epochs in acc_dict.items():
            for epoch in range(num_epochs):
                for emotion in range(num_emotions):
                    mean_accuracies[epoch][emotion] += epochs[epoch][emotion]

        # Divide by the number of chunks to get the mean
        num_chunks = len(acc_dict)
        for epoch in range(num_epochs):
            for emotion in range(num_emotions):
                mean_accuracies[epoch][emotion] /= num_chunks

        return mean_accuracies

    mean_train_accuracies = calculate_mean_accuracies(train_accuracies)
    mean_eval_accuracies = calculate_mean_accuracies(eval_accuracies)
    mean_accuracy_per_batch = [np.mean(epoch) for epoch in
                               [data["eval_accuracy_per_epoch"][batch] for batch in data["eval_accuracy_per_epoch"]]]

    # Plot accuracies per emotion
    epochs = list(range(num_epochs))
    colors = plt.cm.tab10.colors  # Use tab10 colormap for distinct colors

    plt.figure(figsize=(14, 16))  # Increase figure size to accommodate two plots

    # Subplot 1: Per-emotion accuracies
    plt.subplot(2, 1, 1)
    EMOTION_MAP = {
        0: "happy",
        1: "angry",
        2: "sad",
        3: "contemptuous",
        4: "disgusted",
        5: "neutral",
        6: "fearful",
        7: "surprised"
    }

    if plot_train:
        # Plot training accuracies per emotion
        for emotion in range(num_emotions):
            plt.plot(epochs, [mean_train_accuracies[epoch][emotion] for epoch in epochs],
                     label=f'{EMOTION_MAP[emotion]}', color=colors[emotion % len(colors)], linestyle='--')
    else:
        # Plot evaluation accuracies per emotion
        for emotion in range(num_emotions):
            plt.plot(epochs, [mean_eval_accuracies[epoch][emotion] for epoch in epochs],
                     label=f'{EMOTION_MAP[emotion]}', color=colors[emotion % len(colors)], linestyle='-')

    # Extract and format training time
    train_time_seconds = data.get("train_time", 0)
    human_readable_train_time = str(timedelta(seconds=train_time_seconds))

    # Plot styling for first subplot
    plot_type = "Training" if plot_train else "Evaluation"
    plt.title(
        f"Per-Emotion {plot_type} Accuracy (Mean Across Chunks)",
        fontsize=20
    )
    plt.xticks(epochs[::2])
    plt.xlabel("Epoch", fontsize=15)
    plt.yticks(range(0, 110, 10))
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.ylim(0, 100)
    plt.legend(fontsize=23, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Mean accuracy per batch
    plt.subplot(2, 1, 2)
    plt.plot(list(range(len(mean_accuracy_per_batch))), mean_accuracy_per_batch, label="Mean accuracy per batch",
             color=colors[8 % len(colors)],
             linestyle='--', linewidth=2)
    plt.title("Mean Accuracy Per Chunk", fontsize=20)
    plt.xticks(list(range(15))[::2])
    plt.xlabel("Chunk", fontsize=15)
    plt.ylabel("Mean Accuracy (%)", fontsize=15)
    plt.yticks(range(0, 110, 10))
    plt.ylim(0, 100)
    plt.legend(fontsize=23, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot if requested
    if save_plot:
        plt.savefig(f"{name}_plot.png", bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_heatmaps(model_name):
    with open("validation_results.json", "r") as json_file:
        data = json.load(json_file)

    results = None
    for model in data:
        if model["model_name"] == model_name:
            results = model["results"]
    if results is None:
        print(f"Model name: {model_name} not found")
        return

    # Calculate accuracies
    classes = sorted(results.keys(), key=int)  # Ensure sorted order of keys
    accuracies = [results[class_idx][0] / results[class_idx][1] for class_idx in classes]

    # Create a diagonal accuracy matrix
    num_classes = len(classes)
    heatmap_data = np.zeros((num_classes, num_classes))
    np.fill_diagonal(heatmap_data, accuracies)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
        xticklabels=classes, yticklabels=classes
    )
    plt.title(f'Classification Accuracy Heatmap for {model_name}')
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.show()
