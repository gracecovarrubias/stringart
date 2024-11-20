import matplotlib.pyplot as plt
from stringart import StringArtGenerator
import torch

# Initialize generator
generator = StringArtGenerator(nails=100, iterations=1000, weight=20)

# Load and preprocess the image
image_path = "./demo/input/Sample_ML.jpg"
generator.load_image(image_path)
generator.preprocess()
generator.set_seed(0)
generator.set_nails(100)

generator.initialize_rl_model()

# Load the trained model
generator.model.load_state_dict(torch.load("string_art_rl_model.pth"))
generator.model.eval()  # Set to evaluation mode

# Generate string art
print("Generating string art...")
pattern = list(generator.generate_stepwise())

print("Displaying string art...")
# Extract lines for visualization
lines_x = []
lines_y = []
for i, j in zip(pattern, pattern[1:]):
    lines_x.append((i[0], j[0]))
    lines_y.append((i[1], j[1]))

# Visualize the string art pattern
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.xlim(0, generator.data.shape[1])
plt.ylim(0, generator.data.shape[0])
plt.gca().set_aspect("equal")

# Draw lines
for x, y in zip(lines_x, lines_y):
    plt.plot(x, y, color="black", linewidth=0.1)

plt.show()
