import argparse
import os

from PIL import Image
import torch
from torchvision import transforms as T

from model import load_generator


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform():
    return T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def denorm(tensor):
    return (tensor + 1) / 2


def run_inference(input_path, output_path, checkpoint_path):
    device = select_device()
    generator = load_generator(checkpoint_path, device)
    transform = build_transform()

    image = Image.open(input_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    output_tensor = denorm(output_tensor).clamp(0, 1).cpu().squeeze(0)
    output_image = T.ToPILImage()(output_tensor)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Face2Comic inference")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args.input, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
