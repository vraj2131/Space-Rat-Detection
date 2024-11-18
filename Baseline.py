import numpy as np
import math
import random
from collections import deque
from Ship_layout import create_ship_layout

class BaselineBot:
    def __init__(self, map_grid, sensitivity=0.1):
        self.map_grid = map_grid
        self.sensitivity = sensitivity
        self.valid_positions = [(i, j) for i in range(1, map_grid.shape[0] - 1)
                                for j in range(1, map_grid.shape[1] - 1) if map_grid[i, j] == 1]
        self.current_position = random.choice(self.valid_positions)
        self.target_position = self._set_initial_target()
        self.steps_tracker = {"moves": 0, "senses": 0, "detections": 0}
        self.step_count = 0

        self.bot_probabilities = np.zeros(map_grid.shape)
        for i, j in self.valid_positions:
            self.bot_probabilities[i, j] = 1 / len(self.valid_positions)

        self.target_probabilities = np.zeros(map_grid.shape)
        for i, j in self.valid_positions:
            self.target_probabilities[i, j] = 1 / len(self.valid_positions)

        self._determine_initial_location()

        print(f"Starting position: {self.current_position}")
        print(f"Target position: {self.target_position}")
        print(f"Final bot position after setup: {self.current_position}")

        self._hunt_target()

        print("Summary:")
        print(f"Total moves: {self.steps_tracker['moves']}")
        print(f"Blocked Sensory checks: {self.steps_tracker['senses']}")
        print(f"Rat detections: {self.steps_tracker['detections']}")
        print(f"Final bot position: {self.current_position}")
        print(f"Target found at: {self.current_position}")
        print(f"Steps to capture: {self.steps_tracker['moves']}")

    def _set_initial_target(self):
        options = [(i, j) for i, j in self.valid_positions]
        return random.choice(options)

    def _determine_initial_location(self):
        print("Determining Bot Position...")
        potential_positions = set(self.valid_positions)
        while len(potential_positions) > 1:
            sensory_feedback = self._check_surroundings()
            new_positions = {
                (i, j) for (i, j) in potential_positions
                if self._count_obstacles(i, j) == sensory_feedback
            }

            if not new_positions:
                print("All possibilities eliminated. Retaining current guesses.")
                break

            potential_positions = new_positions
            if len(potential_positions) > 1:
                self._random_move()

        if potential_positions:
            self.current_position = next(iter(potential_positions))
        else:
            raise ValueError("Failed to determine location.")
        print("Positioning complete.")
        print("Resolved starting position:", self.current_position)

    def _check_surroundings(self):
        x, y = self.current_position
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        obstacles = sum(
            self.map_grid[x + dx, y + dy] == 0
            for dx, dy in directions
            if 0 <= x + dx < self.map_grid.shape[0] and 0 <= y + dy < self.map_grid.shape[1]
        )
        self.steps_tracker["senses"] += 1
        return obstacles

    def _count_obstacles(self, x, y):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return sum(
            self.map_grid[x + dx, y + dy] == 0
            for dx, dy in directions
            if 0 <= x + dx < self.map_grid.shape[0] and 0 <= y + dy < self.map_grid.shape[1]
        )
        
    def _random_move(self):
        x, y = self.current_position
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        valid_moves = [(dx, dy) for dx, dy in directions if self.map_grid[x + dx, y + dy] == 1]
        if valid_moves:
            self._execute_move(random.choice(valid_moves))

    def _detect_target(self):
        x_bot, y_bot = self.current_position
        x_target, y_target = self.target_position
        dist = abs(x_bot - x_target) + abs(y_bot - y_target)
        ping_chance = math.exp(-self.sensitivity * (dist - 1))
        return random.random() < ping_chance

    def _update_target_probabilities(self, ping):
        updated_probabilities = np.zeros_like(self.target_probabilities)

        for i, j in self.valid_positions:
            dist = abs(i - self.current_position[0]) + abs(j - self.current_position[1])
            ping_chance = math.exp(-self.sensitivity * (dist - 1))
            if ping:
                updated_probabilities[i, j] = self.target_probabilities[i, j] * ping_chance
            else:
                updated_probabilities[i, j] = self.target_probabilities[i, j] * (1 - ping_chance)

        total = updated_probabilities.sum()
        if total > 0:
            updated_probabilities /= total

        self.target_probabilities = updated_probabilities

    def _move_toward_target(self):
        most_likely_target = np.unravel_index(
            np.argmax(self.target_probabilities), self.target_probabilities.shape
        )

        queue = deque([(self.current_position, [])])
        visited = set()

        while queue:
            current, path = queue.popleft()
            if current == most_likely_target:
                if path:
                    self._execute_move(path[0])
                return

            if current in visited:
                continue
            visited.add(current)

            x, y = current
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (x + dx, y + dy)
                if (
                    0 <= neighbor[0] < self.map_grid.shape[0]
                    and 0 <= neighbor[1] < self.map_grid.shape[1]
                    and self.map_grid[neighbor] == 1
                    and neighbor not in visited
                ):
                    queue.append((neighbor, path + [(dx, dy)]))

    def _execute_move(self, direction):
        x, y = self.current_position
        dx, dy = direction
        new_position = (x + dx, y + dy)

        if self.map_grid[new_position] == 1:
            self.current_position = new_position
            self.steps_tracker["moves"] += 1

    def _hunt_target(self):
        time_step = 0
        while not np.isclose(self.target_probabilities.max(), 1.0):
            time_step += 1
            detection = self._detect_target()
            self.steps_tracker["detections"] += 1
            self._update_target_probabilities(detection)

            likely_target = np.unravel_index(
                np.argmax(self.target_probabilities), self.target_probabilities.shape
            )
            print(f"Step {time_step}: Likely target position: {likely_target}")
            self._move_toward_target()

        print("Target captured at:", self.current_position)


if __name__ == "__main__":
    map_grid = create_ship_layout()
    bot = BaselineBot(map_grid, sensitivity=0.1)
