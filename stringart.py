import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import math
import copy
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import bresenham


class StringArtGenerator:
    def __init__(self, nails=100, iterations=1000, weight=20):
        # String Art Parameters
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
        state_size = nails + nails * nails  # Image data + one-hot nail encoding
        self.model = StringArtRLModel(state_size=state_size, action_size=nails)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

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
        assert len(
            self.nodes) == self.nails, "Number of nodes does not match the number of nails!"

    def _get_radius(self):
        return 0.5 * max(self.data.shape)

    def load_image(self, path):
        img = Image.open(path)
        self.image = img
        self.data = np.array(self.image)

    def preprocess(self):
        """Preprocess the input image for string art."""
        self.image = ImageOps.grayscale(self.image)
        self.image = ImageOps.invert(self.image)
        self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        self.image = ImageEnhance.Contrast(self.image).enhance(1)
        self.data = np.array(self.image, dtype=np.float64)
        self.original_data = self.data.copy()  # Save the original for comparison

    def calculate_paths(self):
        """Precompute all paths between nails."""
        self.paths = [
            [bresenham.bresenham_path(
                self.nodes[i], self.nodes[j], self.data.shape) for j in range(self.nails)]
            for i in range(self.nails)
        ]

    def create_state(self, current_nail):
        """Create a flattened state tensor for the RL model."""
        # Flatten the current image state and combine it with the nail's one-hot encoding
        flattened_image = self.data.flatten()
        nail_one_hot = np.zeros(self.nails)
        nail_one_hot[current_nail] = 1
        state = np.concatenate((flattened_image, nail_one_hot))
        return torch.tensor(state, dtype=torch.float32)

    def calculate_reward(self):
        """Calculate the reward as the negative Mean Squared Error (MSE) to the original image."""
        mse = np.mean((self.data - self.original_data) ** 2)
        return -mse

    def choose_next_nail(self, state):
        """Use the RL model to choose the next nail."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state)
            action = torch.argmax(logits).item()
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
        if not self.nodes:
            self._set_nodes()
            self.calculate_paths()
            pattern = []
            nail = self.seed
        self.calculate_paths()
        pattern = []
        nail = self.seed

        for _ in range(self.iterations):
            state = self.create_state(nail)
            next_nail = self.choose_next_nail(state)

            # Update image based on the path from the current nail to the chosen nail
            rows, cols = zip(*self.paths[nail][next_nail])
            self.data[rows, cols] -= self.weight
            self.data[self.data < 0] = 0  # Clamp negative values
            pattern.append(self.nodes[next_nail])

            # Calculate reward and train the model
            reward = self.calculate_reward()
            self.train_model(state, torch.tensor(
                [next_nail]), torch.tensor([reward]))

            # Break if no brightness is left
            if np.sum(self.data) == 0:
                break

            nail = next_nail
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
        return self.fc3(x)  # Output logits for all actions
