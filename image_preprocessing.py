# Image Preprocessing

# Importing necessary libraries
import numpy as np  # For numerical operations and array manipulations
from scipy.misc import imresize  # For resizing images (deprecated; consider `Pillow` or `cv2` for modern usage)
from gym.core import ObservationWrapper  # Gym wrapper to modify environment observations
from gym.spaces.box import Box  # Gym's data structure to define observation space boundaries

# Preprocessing images for the AI
class PreprocessImage(ObservationWrapper):
    """
    This class preprocesses images from the environment by resizing, cropping, 
    normalizing, and optionally converting to grayscale.
    It ensures the images are in the correct format for the AI.
    """

    def __init__(self, env, height=64, width=64, grayscale=True, crop=lambda img: img):
        """
        Initialize the preprocessing wrapper.
        :param env: The Gym environment to wrap.
        :param height: Desired height of the preprocessed images.
        :param width: Desired width of the preprocessed images.
        :param grayscale: Whether to convert images to grayscale.
        :param crop: Optional cropping function to apply to the images.
        """
        super(PreprocessImage, self).__init__(env)  # Initialize the base ObservationWrapper
        self.img_size = (height, width)  # Target image dimensions
        self.grayscale = grayscale  # Whether to convert images to grayscale
        self.crop = crop  # Cropping function (default: no cropping)

        # Define the observation space (boundaries for the image data)
        n_colors = 1 if self.grayscale else 3  # Number of color channels (1 for grayscale, 3 for RGB)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=[n_colors, height, width], dtype=np.float32
        )  # Normalized image values between 0.0 and 1.0

    def _observation(self, img):
        """
        Process an image from the environment.
        :param img: Raw image from the environment.
        :return: Preprocessed image ready for the AI.
        """
        img = self.crop(img)  # Apply cropping function to the image
        img = imresize(img, self.img_size)  # Resize the image to the target dimensions

        if self.grayscale:
            img = img.mean(-1, keepdims=True)  # Convert to grayscale by averaging color channels

        img = np.transpose(img, (2, 0, 1))  # Reorder dimensions to (channels, height, width)
        img = img.astype('float32') / 255.0  # Normalize pixel values to the range [0, 1]
        return img
