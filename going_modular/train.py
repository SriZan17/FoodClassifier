"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import data_setup
import engine
import model_builder
import utils
from torchvision import transforms
import argparse

# Setup default hyperparameters
num_epochs = 5
batch_size = 32
hidden_units = 10
learning_rate = 0.001

parser = argparse.ArgumentParser(description="Train a PyTorch image classifier.")
parser.add_argument(
    "--NUM_EPOCHS",
    type=int,
    help="Number of epochs to train for.",
    default=num_epochs,
)
parser.add_argument(
    "--BATCH_SIZE",
    type=int,
    help="Number of samples per batch.",
    default=batch_size,
)
parser.add_argument(
    "--HIDDEN_UNITS",
    type=int,
    help="Number of hidden units.",
    default=hidden_units,
)
parser.add_argument(
    "--LEARNING_RATE",
    type=float,
    help="Learning rate.",
    default=learning_rate,
)
args = parser.parse_args()
NUM_EPOCHS = args.NUM_EPOCHS
BATCH_SIZE = args.BATCH_SIZE
HIDDEN_UNITS = args.HIDDEN_UNITS
LEARNING_RATE = args.LEARNING_RATE


# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"
# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create transforms
data_transform = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)
# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,
)
# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
).to(device)
# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)
# Save the model with help from utils.py
utils.save_model(
    model=model,
    target_dir="models",
    model_name="05_going_modular_script_mode_tinyvgg_model.pth",
)
