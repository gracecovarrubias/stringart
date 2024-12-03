from stringart import StringArtGenerator
import torch
import config

# Set up the generator
generator = StringArtGenerator(
    nails=config.NAILS, iterations=config.ITERATIONS, weight=config.WEIGHT)
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

# Visualize and evaluate
generator.visualize_pattern(pattern)
