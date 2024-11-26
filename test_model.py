import matplotlib.pyplot as plt
from stringart import StringArtGenerator
import torch

# Set the total number of nails and iterations
total_nails = 25
iterations = 100
weight = 20

# Initialize the StringArtGenerator
generator = StringArtGenerator(
    nails=total_nails, iterations=iterations, weight=weight)

# Load and preprocess the input image
# Replace with the actual path to your image
image_path = "./demo/input/star.jpg"
generator.load_image(image_path)
generator.preprocess()
generator.set_nails(total_nails)

# Initialize and load the trained RL model
generator.initialize_rl_model()
generator.model.load_state_dict(torch.load("string_art_rl_model.pth"))
generator.model.eval()  # Set the model to evaluation mode

# Generate the string art pattern
print("Generating string art...")
pattern = []
for current_nail, next_nail, _ in generator.generate_stepwise():
    pattern.append((current_nail, next_nail))

# Visualize the generated string art pattern
print("Displaying string art...")
lines_x = []
lines_y = []
for current_nail, next_nail in pattern:
    start_node = generator.nodes[current_nail]
    end_node = generator.nodes[next_nail]
    lines_x.append((start_node[0], end_node[0]))
    lines_y.append((start_node[1], end_node[1]))

# Plot the string art pattern
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.xlim(0, generator.data.shape[1])
plt.ylim(0, generator.data.shape[0])
plt.gca().set_aspect("equal")

# Draw lines connecting the nails
for x, y in zip(lines_x, lines_y):
    plt.plot(x, y, color="black", linewidth=0.1)

plt.show()
