import torchvision
import torch
from torch import nn
import data_setup
from pathlib import Path
from timeit import default_timer as timer
import engine
from helper_functions import plot_loss_curves

# from torchvision import transforms


def main():
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # Setup Dirs
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Create a transforms pipeline manually (required for torchvision < 0.13)
    # manual_transforms = transforms.Compose(
    #    [
    #        transforms.Resize(
    #            (224, 224)
    #        ),  # 1. Reshape all images to 224x224 (though some models may require different sizes)
    #        transforms.ToTensor(),  # 2. Turn image values to between 0 & 1
    #        transforms.Normalize(
    #            mean=[
    #                0.485,
    #                0.456,
    #                0.406,
    #            ],  # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
    #            std=[0.229, 0.224, 0.225],
    #        ),  # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
    #    ]
    # )

    # Get a set of pretrained model weights
    weights = (
        torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )  # .DEFAULT = best available weights from pretraining on ImageNet
    # Get the transforms used to create our pretrained weights
    auto_transforms = weights.transforms()

    # Create training and testing DataLoaders as well as get a list of class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=auto_transforms,  # resize, convert images to between 0 & 1 and normalize them
        batch_size=32,
    )  # set mini-batch size to 32

    # Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # Print a summary using torchinfo (uncomment for actual output)
    # w = summary(
    #    model=model,
    #    input_size=(32, 3, 224, 224),  # make sure this is "input_size", not "input_shape"
    #    # col_names=["input_size"], # uncomment for smaller output
    #    col_names=["input_size", "output_size", "num_params", "trainable"],
    #    col_width=20,
    #    row_settings=["var_names"],
    # )
    # print(w)
    print(class_names)
    # Freeze all base layers in the "features" section of the model (the feature extractor)
    # by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=output_shape,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(device)

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Start the timer
    start_time = timer()

    # Setup training and save the results
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=5,
        device=device,
    )

    # End the timer and print out how long it took
    end_time = timer()

    # Save the model
    torch.save(model.state_dict(), "models/efficientnet_b0.pth")

    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    plot_loss_curves(results)


if __name__ == "__main__":
    main()
