from stringart import StringArtGenerator
import torch

total_nails = 25

# Initialize the generator
generator = StringArtGenerator(nails=total_nails, iterations=100, weight=20)

# Load and preprocess the image
image_path = "./demo/input/star.jpg"
generator.load_image(image_path)
generator.preprocess()
generator.set_nails(total_nails)
generator.initialize_rl_model()
# generator.pretrain_with_heuristic()

# Train the RL model
for epoch in range(5):  # Number of training epochs
    print(f"Epoch {epoch + 1}")
    for current_nail, next_nail, path in generator.generate_stepwise():
        print(f"Trained step: Nail {current_nail} -> Nail {next_nail}")

# Save the model
torch.save(generator.model.state_dict(), "string_art_rl_model.pth")
print("Training complete and model saved!")
