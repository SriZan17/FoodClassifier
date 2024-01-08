import torchvision
from model import create_effnetb2_model
from torchvision import datasets
from pathlib import Path
import torch
import os
from going_modular import engine
from going_modular.helper_functions import set_seeds
from going_modular import utils
from going_modular.helper_functions import create_writer


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    NUM_WORKERS = 2 if os.cpu_count() <= 4 else 4
    EPOCHS = 10
    # Set the device globally
    torch.set_default_device(DEVICE)

    # Create EffNetB2 model capable of fitting to 101 classes for Food101
    effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=101)
    effnetb2_food101.to(DEVICE)

    # Create Food101 training data transforms (only perform data augmentation on the training images)
    food101_train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.TrivialAugmentWide(),
            effnetb2_transforms,
        ]
    )

    data_dir = Path("data")

    # Get training data (~750 images x 101 food classes)
    train_data = datasets.Food101(
        root=data_dir,  # path to download data to
        split="train",  # dataset split to get
        transform=food101_train_transforms,  # perform data augmentation on training data
        download=True,
    )  # want to download?
    class_names = train_data.classes  # get class names

    # Get testing data (~250 images x 101 food classes)
    test_data = datasets.Food101(
        root=data_dir,
        split="test",
        transform=effnetb2_transforms,  # perform normal EffNetB2 transforms on test data
        download=True,
    )

    # Create training 20% split of Food101
    # train_data_food101_20_percent, _ = split_dataset(dataset=train_data, split_size=0.2)
    train_data_food101_20_percent = train_data
    # Create testing 20% split of Food101
    # test_data_food101_20_percent, _ = split_dataset(dataset=test_data, split_size=0.2)
    test_data_food101_20_percent = test_data

    len(train_data_food101_20_percent), len(test_data_food101_20_percent)

    # Create Food101 20 percent training DataLoader
    train_dataloader_food101_20_percent = torch.utils.data.DataLoader(
        train_data_food101_20_percent,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # Create Food101 20 percent testing DataLoader
    test_dataloader_food101_20_percent = torch.utils.data.DataLoader(
        test_data_food101_20_percent,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(params=effnetb2_food101.parameters(), lr=1e-3)

    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=0.1
    )  # throw in a little label smoothing because so many classes

    # Setup the PyTorch TensorBoard logger
    writer = create_writer(
        experiment_name="20%-data-" + str(EPOCHS) + "-epochs",
        model_name="effnetb2",
        extra=f"{EPOCHS}_epochs",
    )

    # Want to beat original Food101 paper with 20% of data, need 56.4%+ acc on test dataset
    set_seeds()
    effnetb2_food101_results = engine.train(
        model=effnetb2_food101,
        train_dataloader=train_dataloader_food101_20_percent,
        test_dataloader=test_dataloader_food101_20_percent,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=DEVICE,
        writer=writer,
    )

    # Create a model path
    effnetb2_food101_model_path = "effnetb2_food101.pth"

    # Save FoodVision Big model
    utils.save_model(
        model=effnetb2_food101,
        target_dir="models",
        model_name=effnetb2_food101_model_path,
    )

    # Create path to Food101 class names

    foodvision_big_class_names_path = "class_names.txt"

    # Write Food101 class names list to file
    with open(foodvision_big_class_names_path, "w") as f:
        print(f"[INFO] Saving Food101 class names to {foodvision_big_class_names_path}")
        f.write("\n".join(class_names))  # leave a new line between each class


def split_dataset(
    dataset: torchvision.datasets, split_size: float = 0.2, seed: int = 42
):
    """Randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
        split_size (float, optional): How much of the dataset should be split?
            E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
        seed (int, optional): Seed for random generator. Defaults to 42.

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and
            random_split_2 is of size (1-split_size)*len(dataset).
    """
    # Create split lengths based on original dataset length
    length_1 = int(len(dataset) * split_size)  # desired length
    length_2 = len(dataset) - length_1  # remaining length

    # Print out info
    print(
        f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%)"
    )
    print(f" {length_2} ({int((1-split_size)*100)}%)")

    # Create splits with given random seed
    random_split_1, random_split_2 = torch.utils.data.random_split(
        dataset, lengths=[length_1, length_2], generator=torch.manual_seed(seed)
    )  # set the random seed for reproducible splits
    return random_split_1, random_split_2


if __name__ == "__main__":
    main()
