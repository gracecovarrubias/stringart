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

# Train for multiple epochs
epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    pattern = generator.generate()

# Save the model
torch.save(generator.model.state_dict(), "string_art_model.pth")
print("Training complete. Model saved.")
