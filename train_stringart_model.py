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

# Train for multiple epochs
epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    pattern = generator.generate()

# Save the model
torch.save(generator.model.state_dict(), "string_art_model.pth")
print("Training complete. Model saved.")
