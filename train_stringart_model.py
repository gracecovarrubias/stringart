import matplotlib.pyplot as plt
from stringart import StringArtGenerator
import torch
import config
from evaluate import evaluate_model

# Set up the generator
generator = StringArtGenerator(
    nails=config.NAILS, iterations=config.ITERATIONS, weight=config.WEIGHT)
generator.load_image(config.IMAGE_PATH)
generator.preprocess()
generator.set_nails(config.NAILS)
generator.initialize_rl_model()

# Train for multiple epochs
epochs = 10

mse_values = []

for epoch in range(epochs):
    pattern = generator.generate()
    mse, _ = evaluate_model(generator, pattern)
    mse_values.append(mse)

# Plot MSE over epochs
plt.plot(mse_values)
plt.title("Training Progress")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# Save the model
torch.save(generator.model.state_dict(), "string_art_model.pth")
print("Training complete. Model saved.")
