from stringart import StringArtGenerator
import torch

image_path = "./demo/input/star.jpg"
nail_total = 200
iterations = 2000

# Set up the generator
generator = StringArtGenerator(
    nails=nail_total, iterations=iterations, weight=20)
generator.load_image(image_path)
generator.preprocess()
generator.set_nails(nail_total)
generator.initialize_rl_model()

# Load the trained model
generator.model.load_state_dict(torch.load("string_art_model.pth"))
generator.model.eval()
print("Loaded trained model.")

# Generate string art pattern
pattern = generator.generate()

# Visualize and evaluate
generator.visualize_pattern(pattern)
