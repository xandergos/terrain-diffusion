import torch
from torchvision import transforms as T

class TupleTransform(torch.nn.Module):
    def __init__(self, *transforms):
        """
        Initialize TupleTransform with multiple transforms.
        
        Args:
            *transforms: Variable number of transform objects to be applied.
        """
        super().__init__()
        self.transforms = transforms

    def forward(self, img):
        """
        Apply all transforms to the input image and return results as a tuple.
        
        Args:
            img: Input image to be transformed.
        
        Returns:
            tuple: Results of applying each transform to the input image.
        """
        return tuple(t(img) for t in self.transforms)
