from utils import calculate_confusion_matrix
import torch
import torchvision
from mlxtend.plotting import plot_confusion_matrix


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transform = weights.transforms()
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=3,  # same number of output units as our number of classes
            bias=True,
        ),
    ).to(device)
    model.load_state_dict(torch.load("models/efficientnet_b0.pth"))
    class_names = ["Pizza", "Steak", "Sushi"]
    test_dir = "data/pizza_steak_sushi/test"

    confmat_tensor = calculate_confusion_matrix(
        class_names=class_names,
        model=model,
        test_dir=test_dir,
        device=device,
        transform=transform,
    )
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor,  # matplotlib likes working with NumPy
        class_names=class_names,  # turn the row and column labels into class names
        figsize=(10, 7),
    )
    ax.set_xlabel("Predicted label", fontsize=15)
    fig.show()
    fig.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
