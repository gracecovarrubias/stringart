from stringart import StringArtGenerator
import torch

# Initialize the generator
generator = StringArtGenerator(nails=100, iterations=1000, weight=20)

# Load and preprocess the image
image_path = "./demo/input/Sample_ML.jpg"
generator.load_image(image_path)
generator.preprocess()
generator.set_seed(0)
generator.set_nails(100)
generator.initialize_rl_model()

# Train the RL model
print("Starting training...")
for epoch in range(5):  # Number of training epochs
    print(f"Epoch {epoch + 1}")
    for step in generator.generate_stepwise():
        pass  # Training happens within generate_stepwise

# Save the model
torch.save(generator.model.state_dict(), "string_art_rl_model.pth")
print("Training complete and model saved!")
