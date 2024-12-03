import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import math
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import torch.nn.functional as F

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
                path.append((y1, x1))  # Store coordinates as (y, x)
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
        # Create a copy of the current data
        updated_data = self.data.copy()

        # Apply the updates from the Bresenham path
        for y, x in after_update:
            if 0 <= y < updated_data.shape[0] and 0 <= x < updated_data.shape[1]:
                updated_data[y, x] -= self.weight
                if updated_data[y, x] < 0:
                    updated_data[y, x] = 0

        # Calculate MSE for the current and updated states
        mse_current = np.mean((self.data - self.original_data) ** 2)
        mse_new = np.mean((updated_data - self.original_data) ** 2)

        return mse_current - mse_new  # Reward is the improvement in MSE

    def choose_next_nail(self, state, epsilon=0.1):
        """
        Choose the next nail using epsilon-greedy policy.
        """
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randint(0, self.nails - 1)
        else:
            # Exploitation: choose the best action based on Q-values
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def train_model(self, state, action, reward, next_state, done, gamma=0.99):
        """
        Train the RL model using Q-learning updates.
        """
        # Get the predicted Q-values for the current state
        q_values = self.model(state)

        # Compute the target Q-value
        with torch.no_grad():
            q_next_values = self.model(next_state)
            max_q_next = torch.max(q_next_values).item() if not done else 0.0
            target_q_value = reward + gamma * max_q_next

        # Update the Q-value for the selected action
        target_q_values = q_values.clone().detach()
        target_q_values[action] = target_q_value

        # Compute the loss and backpropagate
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generate(self, epsilon=0.1):
        """
        Generate the string art pattern using RL.
        """
        current_nail = 0
        state = self.create_state(current_nail)
        pattern = []

        for _ in range(self.iterations):
            action = self.choose_next_nail(state, epsilon)
            next_nail = action
            self.paths.append((current_nail, next_nail))

            # Simulate the environment transition
            # updated_image = self.bresenham_path(current_nail, next_nail)

            # Simulate the transition using Bresenham's path
            updated_image = self.bresenham_path(
                self.nodes[current_nail], self.nodes[next_nail])

            # Calculate the reward for the action
            reward = self.calculate_reward(updated_image)

            # Apply the updates from Bresenham to the actual data
            for y, x in updated_image:
                if 0 <= y < self.data.shape[0] and 0 <= x < self.data.shape[1]:
                    self.data[y, x] -= self.weight
                    if self.data[y, x] < 0:
                        self.data[y, x] = 0

            next_state = self.create_state(next_nail)
            done = False  # Define termination condition if needed

            # Train the RL model
            self.train_model(state, action, reward, next_state, done)

            # Update state and nail
            state = next_state
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
