import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import math
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class StringArtGenerator:
    def __init__(self, nails=100, iterations=1000, weight=20):
        self.iterations = iterations
        self.nails = nails
        self.weight = weight
        self.seed = 0
        self.image = None
        self.data = None
        self.original_data = None
        self.nodes = []
        self.paths = []

        # RL Model
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()

    def initialize_rl_model(self):
        """Initialize the RL model after loading the image."""
        image_size = self.data.size  # Total number of pixels in the image
        state_size = image_size + self.nails  # Image data + one-hot nail encoding
        self.model = StringArtRLModel(
            state_size=state_size, action_size=self.nails)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print("RL model initialized with random weights.")

    def set_seed(self, seed):
        self.seed = seed

    def set_nails(self, nails):
        self.nails = nails
        self._set_nodes()

    def _set_nodes(self):
        """Sets nails evenly along a circle of given diameter."""
        spacing = (2 * math.pi) / self.nails
        radius = self._get_radius()
        self.nodes = [
            (radius + radius * math.cos(i * spacing),
             radius + radius * math.sin(i * spacing))
            for i in range(self.nails)
        ]

    def _get_radius(self):
        return 0.5 * max(self.data.shape)

    def load_image(self, path):
        img = Image.open(path)
        self.image = img
        self.data = np.array(self.image, dtype=np.float64)

    def preprocess(self):
        """Preprocess the input image for string art."""
        self.image = ImageOps.grayscale(self.image)
        self.image = ImageOps.invert(self.image)
        self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        self.image = ImageEnhance.Contrast(self.image).enhance(1)
        self.data = np.array(self.image, dtype=np.float64)
        self.original_data = self.data.copy()

    def calculate_paths(self):
        """Precompute all paths between nails."""
        self.paths = []
        for i in range(self.nails):
            self.paths.append([])
            for j in range(self.nails):
                if i == j:
                    self.paths[i].append([])  # No path for the same nail
                    continue
                path = self.interpolate_line(self.nodes[i], self.nodes[j])
                self.paths[i].append(path)

    def interpolate_line(self, start, end):
        """Generate points along a line using interpolation."""
        x1, y1 = start
        x2, y2 = end

        # Ensure sufficient resolution
        num_points = int(max(abs(x2 - x1), abs(y2 - y1)))
        x_vals = np.linspace(x1, x2, num_points).astype(int)
        y_vals = np.linspace(y1, y2, num_points).astype(int)

        path = list(zip(y_vals, x_vals))  # (row, col) format
        return [(y, x) for y, x in path if 0 <= y < self.data.shape[0] and 0 <= x < self.data.shape[1]]

    def create_state(self, current_nail):
        """Create a flattened state tensor for the RL model."""
        # Flatten the current image state
        flattened_image = self.data.flatten()

        # Create a one-hot encoding for the current nail
        nail_one_hot = np.zeros(self.nails, dtype=np.float32)
        nail_one_hot[current_nail] = 1.0

        # Combine image data and one-hot encoding into a single state
        state = np.concatenate((flattened_image, nail_one_hot))

        return torch.tensor(state, dtype=torch.float32)

    def calculate_reward(self, before_update, after_update):
        """Calculate the reward as the local improvement to the image."""
        mse_before = np.mean((before_update - self.original_data) ** 2)
        mse_after = np.mean((after_update - self.original_data) ** 2)
        return mse_before - mse_after  # Positive reward for improvement

    def choose_next_nail(self, state, current_nail):
        """Use the RL model to choose the next nail."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state)

            # Mask invalid actions (e.g., selecting the current nail)
            mask = torch.ones(self.nails, dtype=torch.bool)
            mask[current_nail] = False  # Mask out the current nail
            # Assign negative infinity to invalid actions
            logits[~mask] = float('-inf')

            probabilities = torch.softmax(logits, dim=-1)
            action = torch.multinomial(
                probabilities, 1).item()  # Sample action
        return action

    def train_model(self, state, action, reward):
        """Train the RL model using the reward signal."""
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(state)
        log_prob = torch.log_softmax(logits, dim=-1)[action]
        loss = -log_prob * reward  # Policy gradient loss
        loss.backward()
        self.optimizer.step()

    def generate_stepwise(self):
        """Generate pattern step by step and train the RL model."""
        self.calculate_paths()
        pattern = []
        current_nail = self.seed  # Start at the seed nail

        for _ in range(self.iterations):
            # Create the state based on the current nail
            state = self.create_state(current_nail)

            # Choose the next nail
            next_nail = self.choose_next_nail(state, current_nail)

            # Ensure the path is valid
            path = self.paths[current_nail][next_nail]
            if not path:
                print(f"No valid path from nail {
                    current_nail} to nail {next_nail}. Skipping.")
                continue

            # Save the state before updating
            before_update = self.data.copy()

            # Update the image based on the path
            rows, cols = zip(*path)
            self.data[rows, cols] -= self.weight
            self.data[self.data < 0] = 0  # Clamp negative values
            pattern.append(self.nodes[next_nail])

            # Calculate reward based on improvement
            reward = self.calculate_reward(before_update, self.data)
            self.train_model(state, torch.tensor(
                [next_nail]), torch.tensor([reward]))

            # Break if no brightness is left
            if np.sum(self.data) == 0:
                break

            # Update the current nail to the newly selected nail
            current_nail = next_nail

            yield self.nodes[next_nail]  # Yield progress

        return pattern


class StringArtRLModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(StringArtRLModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Logits for the action space
