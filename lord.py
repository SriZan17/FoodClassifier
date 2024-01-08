import torch
import torchvision
from torch import nn
from pathlib import Path
from going_modular import data_setup
from going_modular import utils
from going_modular import engine
from going_modular import helper_functions
import json

# from torchinfo import summary


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 10

    # Download pizza, steak, sushi images from GitHub
    data_20_percent_path = helper_functions.download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
        destination="pizza_steak_sushi_20_percent",
    )

    # Setup directory paths to train and test images
    train_dir = data_20_percent_path / "train"
    test_dir = data_20_percent_path / "test"
    effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3, seed=43)
    (
        train_dataloader_effnetb2,
        test_dataloader_effnetb2,
        class_names,
    ) = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=effnetb2_transforms,
        batch_size=1024,
    )
    # Setup optimizer

    optimizer = torch.optim.Adam(params=effnetb2.parameters(), lr=1e-3)
    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup the PyTorch TensorBoard logger
    writer = helper_functions.create_writer(
        experiment_name="20%-data-10-epochs",
        model_name="effnetb2",
        extra=f"{EPOCHS}_epochs",
    )

    # Set seeds for reproducibility and train the model
    helper_functions.set_seeds()
    effnetb2_results = engine.train(
        model=effnetb2.to(DEVICE),
        train_dataloader=train_dataloader_effnetb2,
        test_dataloader=test_dataloader_effnetb2,
        epochs=EPOCHS,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        writer=writer,
    )
    # Save model
    utils.save_model(
        model=effnetb2, model_name="effnetb2" + ".pth", target_dir="models"
    )
    # Count number of parameters in EffNetB2
    effnetb2_total_params = sum(torch.numel(param) for param in effnetb2.parameters())
    # Get the model size in bytes then convert to megabytes
    pretrained_effnetb2_model_size = Path("models/effnetb2.pth").stat().st_size // (
        1024 * 1024
    )  # division converts bytes to megabytes (roughly)
    print(
        f"Pretrained EffNetB2 feature extractor model size: {pretrained_effnetb2_model_size} MB"
    )
    # Create a dictionary with EffNetB2 statistics
    effnetb2_stats = {
        "test_loss": effnetb2_results["test_loss"][-1],
        "test_acc": effnetb2_results["test_acc"][-1],
        "number_of_parameters": effnetb2_total_params,
        "model_size (MB)": pretrained_effnetb2_model_size,
    }
    # Save the dictionary as a JSON file
    with open("effnetb2_stats.json", "w") as json_file:
        json.dump(effnetb2_stats, json_file)

    helper_functions.plot_loss_curves(effnetb2_results)


def create_effnetb2_model(num_classes: int = 3, seed: int = 42):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model.
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # 4. Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # 5. Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return model, transforms


if __name__ == "__main__":
    main()
