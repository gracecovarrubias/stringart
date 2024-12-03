from stringart import StringArtGenerator
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import config


def evaluate_model(generator, pattern):
    """Evaluate the performance of the trained model."""
    final_image = generator.data
    mse = np.mean((final_image - generator.original_data) ** 2)
    ssim_value = ssim(
        generator.original_data,
        final_image,
        data_range=generator.original_data.max() - generator.original_data.min()
    )
    print(f"Evaluation Results:\n - MSE: {mse}\n - SSIM: {ssim_value}")
    return mse, ssim_value


if __name__ == "__main__":
    # Create and set up the generator
    generator = StringArtGenerator(
        nails=config.NAILS, iterations=config.ITERATIONS, weight=config.WEIGHT)
    # Change to your test image path
    generator.load_image(config.IMAGE_PATH)
    generator.preprocess()
    generator.set_nails(config.NAILS)
    generator.initialize_rl_model()

    # Load the trained model
    generator.model.load_state_dict(torch.load("string_art_model.pth"))
    generator.model.eval()
    print("Loaded trained model.")

    # Generate string art pattern
    pattern = generator.generate()

    # Evaluate the generated pattern
    evaluate_model(generator, pattern)
