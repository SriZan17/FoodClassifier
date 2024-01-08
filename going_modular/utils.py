"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from torchmetrics import ConfusionMatrix
from tqdm.auto import tqdm
import os
from torchvision import datasets


def calculate_confusion_matrix(
    class_names: list,
    model,  # y_true: torch.Tensor, y_pred: torch.Tensor
    test_dir: str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    transform=None,
):
    """Calculates a PyTorch confusion matrix using the ConfusionMatrix class from torchmetrics.

    Args:
      class_names: A list of class names in the order of the confusion matrix.
      y_true: A tensor of true labels.
      y_pred: A tensor of predicted labels.

    Returns:
      A PyTorch confusion matrix.
    """

    y_preds = []
    test_dir = "data/pizza_steak_sushi/test"
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=False, num_workers=os.cpu_count() - 1
    )

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(
                dim=1
            )  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")

    confmat_tensor = confmat(
        preds=y_pred_tensor, target=torch.tensor(test_data.targets)
    )
    return confmat_tensor.numpy()


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
