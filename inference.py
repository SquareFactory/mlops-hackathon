import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from training import Classifier


class Predictor:
    """Single model predictor."""

    def __init__(self, checkpoint_path: str, force_cpu: bool = False):
        """Initialize the model from a checkpoint."""
        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        print(
            f"Loading model from checkpoint: {checkpoint_path} \n to device: {self.device}"
        )
        self.model = Classifier.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()
        self.model.freeze()
        self.model.to(self.device)

        self.tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    @torch.no_grad()
    def __call__(self, imgs: list):
        """Forward pass for a batch."""
        tensors = []
        for img in imgs:
            if not isinstance(img, np.ndarray):
                raise TypeError("all images must be numpy.ndarray")
            if not len(img.shape) == 2:
                raise TypeError("all images must have the shape [H, W]")
            tensors.append(self.tf(img))
        tensors = torch.stack(tensors, dim=0).to(self.device)

        outputs = self.model(tensors).softmax(dim=-1)

        return outputs.cpu()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise RuntimeError("Invalid path specified")
    mnistmodel = Predictor(args.checkpoint)
    image = cv2.imread(args.image)
    if len(image.shape) == 3:
        image = image[0]  # only take first channel
    outputs = mnistmodel([image])
    print(
        f"Inference results for {args.image}:\n {outputs} \n Prediction result:\n {np.argmax(outputs)+1}"
    )
