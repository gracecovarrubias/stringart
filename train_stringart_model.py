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
epochs = 2

mse_values = []
ssim_values = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    pattern = generator.generate()
    mse, ssim_score = evaluate_model(generator, pattern)
    mse_values.append(mse)
    ssim_values.append(ssim_score)

# Plot metrics
plt.figure(figsize=(8, 6))
plt.plot(mse_values, label="MSE", marker='o')
plt.plot(ssim_values, label="SSIM", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Metrics Over Epochs")
plt.legend()
plt.grid(True)

# Save the plot as an image
output_plot_path = "training_metrics.png"
plt.savefig(output_plot_path, bbox_inches="tight", dpi=300)
print(f"Metrics plot saved to {output_plot_path}")

# Display the plot
plt.show()

# Save the model
torch.save(generator.model.state_dict(), "string_art_model.pth")
print("Training complete. Model saved.")
