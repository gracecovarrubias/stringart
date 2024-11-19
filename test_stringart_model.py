import matplotlib.pyplot as plt
import torch

from stringart import StringArtGenerator

# Initialize the generator
generator = StringArtGenerator(nails=100, iterations=1000, weight=20)

# Load and preprocess the image
image_path = "./demo/input/Sample_ML.jpg"
generator.load_image(image_path)
generator.preprocess()
generator.set_seed(0)

# Train the RL model
print("Starting training...")
for epoch in range(1):  # Number of training epochs
    print(f"Epoch {epoch + 1}")
    for step in generator.generate_stepwise():
        pass  # Training happens within generate_stepwise

print("Training complete!")

# Save the model
model_path = "string_art_rl_model.pth"
torch.save(generator.model.state_dict(), model_path)
print(f"Model saved to {model_path}!")

# Generate and display final pattern
print("Generating final pattern...")
pattern = list(generator.generate_stepwise())
print("Pattern generated!")

# Visualize the pattern

# Extract lines for visualization
lines_x = []
lines_y = []
for i, j in zip(pattern, pattern[1:]):
    lines_x.append((i[0], j[0]))
    lines_y.append((i[1], j[1]))

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.xlim(0, generator.data.shape[1])
plt.ylim(0, generator.data.shape[0])
plt.gca().set_aspect("equal")

# Draw lines
for x, y in zip(lines_x, lines_y):
    plt.plot(x, y, color="black", linewidth=0.1)

plt.show()
