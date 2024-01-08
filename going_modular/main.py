import torch
import torchvision
from helper_functions import pred_and_plot_image


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ["Pizza", "Steak", "Sushi"]
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
    Prediction = pred_and_plot_image(
        model=model,
        class_names=class_names,
        device=device,
        transform=transform,
        image_path="data/pizza_steak_sushi/test/sushi/46797.jpg",
    )
    predicted_label = Prediction.label
    predicted_prob = Prediction.confidence
    print(f"Predicted label: {predicted_label}")
    print(f"Predicted probability: {predicted_prob}")


if __name__ == "__main__":
    main()
