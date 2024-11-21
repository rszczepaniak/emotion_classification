import json

from matplotlib import pyplot as plt


def insert_data_into_json(dataset_name: str, plot_name: str, num_epochs: int, train_acc: list, eval_acc: list,
                          json_data_path: str = "models_training_results.json") -> None:
    with open(json_data_path, "r") as json_file:
        data = json.load(json_file)

    data.append({
        "dataset_name": dataset_name,
        "name": plot_name,
        "num_epochs": num_epochs,
        "train_accuracy_per_epoch": train_acc,
        "eval_accuracy_per_epoch": eval_acc
    })
    with open(json_data_path, "w") as json_file:
        json.dump(data, json_file)


def plot_all_models(dataset_name: str, num_epochs: int, name_prefix: str, applied_transformation: str = "") -> None:
    with open("models_training_results.json", "r") as json_file:
        data = json.load(json_file)
    colors = ("orange", "blue", "green", "red", "purple", "cyan", "yellow")
    color_id = 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plt.title(name_prefix)
    # Plot training accuracy
    axes[0].set_title('Training Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 100)

    # Plot evaluation accuracy
    axes[1].set_title('Evaluation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylim(0, 100)

    for trained_model in data:
        if trained_model["dataset_name"] == dataset_name and trained_model["num_epochs"] == num_epochs and trained_model["name"].startswith(name_prefix) and applied_transformation in trained_model["name"]:
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
