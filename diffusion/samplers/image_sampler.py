import torch
from diffusion.samplers.sampler import Sampler

class ImageSampler(Sampler):
    """
    A simple sampler that retrieves regions from an input image tensor.
    """

    def __init__(self, image: torch.Tensor, translate_x: int = 0, translate_y: int = 0):
        """
        Initialize the ImageSampler with an input image tensor.

        Args:
            image (torch.Tensor): The input image as a PyTorch tensor.
            translate_x (int, optional): Number of pixels to translate the image horizontally. Default is 0.
            translate_y (int, optional): Number of pixels to translate the image vertically. Default is 0.
        """
        self.image = image
        self.translate_x = translate_x
        self.translate_y = translate_y

    def get_region(self, top: int, left: int, bottom: int, right: int, generate=True,
                   enforce_bounds=True):
        """
        Get a region of the image.

        Args:
            top (int): The top coordinate of the region to retrieve.
            left (int): The left coordinate of the region to retrieve.
            bottom (int): The bottom coordinate of the region to retrieve.
            right (int): The right coordinate of the region to retrieve.
            generate (bool, optional): Not used in this implementation.
            enforce_bounds (bool, optional): Whether to throw an error if the region is out of bounds.
        Returns:
            torch.Tensor: A tensor containing the requested region of the image.
        """
        # Apply translation
        top -= self.translate_y
        bottom -= self.translate_y
        left -= self.translate_x
        right -= self.translate_x

        # Check if the requested region is out of bounds (even partially)
        if enforce_bounds and (top < 0 or left < 0 or bottom > self.image.shape[-2] or right > self.image.shape[-1]):
            raise ValueError(f"Requested region ({top}, {left}, {bottom}, {right}) is out of bounds. Image shape is {self.image.shape[-2]}x{self.image.shape[-1]}.")
        
        # Get the dimensions of the requested region
        region_height = bottom - top
        region_width = right - left

        # Create a zero-padded output tensor
        output = torch.zeros(*self.image.shape[:-2], region_height, region_width, device=self.image.device)

        # Calculate the valid region within the image bounds
        valid_top = max(0, top)
        valid_bottom = min(self.image.shape[-2], bottom)
        valid_left = max(0, left)
        valid_right = min(self.image.shape[-1], right)

        # Calculate the corresponding region in the output tensor
        out_top = valid_top - top
        out_bottom = out_top + (valid_bottom - valid_top)
        out_left = valid_left - left
        out_right = out_left + (valid_right - valid_left)

        # Copy the valid region from the image to the output tensor
        output[..., out_top:out_bottom, out_left:out_right] = self.image[..., valid_top:valid_bottom, valid_left:valid_right]

        return output
