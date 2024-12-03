import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import math
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class StringArtGenerator:
    def __init__(self, nails=100, iterations=1000, weight=20):
        self.nails = nails
        self.iterations = iterations
        self.weight = weight
        self.seed = 0
        self.image = None
        self.data = None
        self.original_data = None
        self.nodes = []
        self.paths = []
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()

    def initialize_rl_model(self):
        """Initialize the RL model."""
        image_size = self.data.size
        state_size = image_size + self.nails
        self.model = StringArtRLModel(
            state_size=state_size, action_size=self.nails).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print("RL model initialized on", device)

    def preprocess(self):
        """Preprocess the input image."""
        self.image = ImageOps.grayscale(self.image)
        self.image = ImageOps.invert(self.image)
        self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        self.image = ImageEnhance.Contrast(self.image).enhance(1)
        self.data = np.array(self.image, dtype=np.float64)
        self.original_data = self.data.copy()

    def set_nails(self, nails):
        self.nails = nails
        self._set_nodes()

    def _set_nodes(self):
        """Set nails evenly along a circle."""
        radius = self._get_radius()
        angle_step = 2 * math.pi / self.nails
        self.nodes = [
            (
                radius + radius * math.cos(i * angle_step),
                radius + radius * math.sin(i * angle_step),
            )
            for i in range(self.nails)
        ]

    def _get_radius(self):
        return min(self.data.shape) // 2

    def load_image(self, path):
        """Load and process an image."""
        img = Image.open(path)
        self.image = img
        self.data = np.array(self.image, dtype=np.float64)

    def calculate_paths(self):
        """Precompute paths between all pairs of nails."""
        self.paths = [
            [self.bresenham_path(self.nodes[i], self.nodes[j])
             for j in range(self.nails)]
            for i in range(self.nails)
        ]

    def bresenham_path(self, start, end):
        """Bresenham's Line Algorithm for generating pixel paths."""
        x1, y1 = map(int, start)
        x2, y2 = map(int, end)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        path = []

        while True:
            if 0 <= y1 < self.data.shape[0] and 0 <= x1 < self.data.shape[1]:
                path.append((y1, x1))
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return path

    def create_state(self, current_nail):
        """Create a flattened state tensor for the RL model."""
        flattened_image = self.data.flatten()
        nail_one_hot = np.zeros(self.nails, dtype=np.float32)
        nail_one_hot[current_nail] = 1.0
        state = np.concatenate((flattened_image, nail_one_hot))
        return torch.tensor(state, dtype=torch.float32).to(device)

    def calculate_reward(self, after_update):
        """Calculate reward based on MSE improvement."""
        mse_current = np.mean((self.data - self.original_data) ** 2)
        mse_new = np.mean((after_update - self.original_data) ** 2)
        return mse_current - mse_new  # Reward is the improvement in MSE

    def choose_next_nail(self, state):
        """Use the RL model to predict the next nail."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(state)
            probabilities = torch.softmax(logits, dim=-1)
            # Sample from probabilities
            next_nail = torch.multinomial(probabilities, 1).item()
        return next_nail

    def train_model(self, state, action, reward):
        """Train the RL model using the reward signal."""
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(state)
        log_prob = torch.log_softmax(logits, dim=-1)[action]
        loss = -log_prob * reward
        loss.backward()
        self.optimizer.step()

    def generate(self):
        """Generate the string art pattern using RL."""
        self.calculate_paths()
        current_nail = self.seed
        pattern = []

        for _ in range(self.iterations):
            state = self.create_state(current_nail)
            next_nail = self.choose_next_nail(state)
            path = self.paths[current_nail][next_nail]

            if not path:
                print(f"No valid path from nail {
                      current_nail} to nail {next_nail}. Skipping.")
                continue

            before_update = self.data.copy()
            rows, cols = zip(*path)
            self.data[rows, cols] -= self.weight
            self.data[self.data < 0] = 0

            reward = self.calculate_reward(self.data)
            self.train_model(state, next_nail, reward)

            pattern.append((current_nail, next_nail))
            current_nail = next_nail

        return pattern

    def visualize_pattern(self, pattern):
        """Visualize the generated string art pattern."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.imshow(self.original_data, cmap="gray")
        plt.axis("off")

        for start, end in pattern:
            start_node = self.nodes[start]
            end_node = self.nodes[end]
            plt.plot(
                [start_node[0], end_node[0]], [start_node[1], end_node[1]], "r-", alpha=0.6
            )

        plt.show()


class StringArtRLModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(StringArtRLModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
