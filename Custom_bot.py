import numpy as np
import random
import math
from collections import deque
from Ship_layout import create_ship_layout


class ParticleFilter:
    def __init__(self, grid, num_particles=100, alpha=0.1, threshold=0.5, decay=0.9):
        self.grid = grid
        self.num_particles = num_particles
        self.alpha = alpha 
        self.threshold = threshold  
        self.decay = decay  
        self.particles = [] 

    def initialize_particles(self, rat_position=None, bot_position=None):
        particles = []
        possible_positions = [
            (x, y)
            for x in range(1, self.grid.shape[0] - 1)
            for y in range(1, self.grid.shape[1] - 1)
            if self.grid[x, y] == 1  
        ]
        
        if rat_position:  
            weighted_positions = self.weighted_positions(rat_position, possible_positions)
        elif bot_position:  
            weighted_positions = self.weighted_positions(bot_position, possible_positions)
        else:
            weighted_positions = possible_positions  

        for _ in range(self.num_particles):
            particle = random.choice(weighted_positions)
            particles.append({'position': particle, 'weight': 1.0})

        self.particles = particles

    def weighted_positions(self, target_position, possible_positions):
        x_target, y_target = target_position
        weighted_positions = []
        for pos in possible_positions:
            x, y = pos
            distance = abs(x - x_target) + abs(y - y_target)  
            weight = 1 / (distance + 1)  
            weighted_positions.extend([pos] * int(weight * 10))  

        return weighted_positions

    def predict(self, bot_position, ping_result):
        total_weight = 0
        for particle in self.particles:
            x_rat, y_rat = particle['position']
            x_bot, y_bot = bot_position
            distance = abs(x_rat - x_bot) + abs(y_rat - y_bot)
            prob_ping = math.exp(-self.alpha * (distance - 1))

            if ping_result:
                particle['weight'] *= prob_ping
            else:
                particle['weight'] *= (1 - prob_ping)

            if not np.isfinite(particle['weight']) or particle['weight'] <= 0:
                particle['weight'] = 1e-6  
            total_weight += particle['weight']

            x_move = random.choice([-1, 0, 1])  
            y_move = random.choice([-1, 0, 1])  

            new_x = max(1, min(self.grid.shape[0] - 2, x_rat + x_move))  
            new_y = max(1, min(self.grid.shape[1] - 2, y_rat + y_move))  
            if self.grid[new_x, new_y] == 1:  
                particle['position'] = (new_x, new_y)

        if total_weight == 0:
            print("Warning: All particles have zero weight, reinitializing particles.")
            self.initialize_particles()

    def resample(self):
        total_weight = sum(p['weight'] for p in self.particles)

        if total_weight <= 0:
            print("Warning: Total weight is zero or invalid. Reinitializing particles.")
            self.particles = self.initialize_particles()  
            total_weight = sum(p['weight'] for p in self.particles)  

        total_weight = sum(p['weight'] for p in self.particles)
        normalized_particles = []
        for p in self.particles:
            p['weight'] /= total_weight if total_weight > 0 else 1
            normalized_particles.append(p)

        total_weight = sum(p['weight'] for p in normalized_particles)
        if total_weight <= 0:
            print("Warning: Total weight after normalization is invalid. Reinitializing particles.")
            self.particles = self.initialize_particles()
        else:
            self.particles = random.choices(
                normalized_particles, weights=[p['weight'] for p in normalized_particles], k=self.num_particles
            )

    def estimate_rat_position(self):
        weighted_positions = np.array([p['position'] for p in self.particles])
        weights = np.array([p['weight'] for p in self.particles])
        weighted_sum = np.sum(weighted_positions * weights[:, None], axis=0)
        return tuple(weighted_sum / np.sum(weights))  



class CustomBot:
    def __init__(self, map_grid, alpha=0.1, threshold=0.5, decay=0.9, oscillation_limit=3, grid_size=20, num_particles=100):
        self.map_grid = map_grid
        self.alpha = alpha
        self.threshold = threshold
        self.decay = decay
        self.oscillation_limit = oscillation_limit
        self.grid_size = grid_size  
        self.num_particles = num_particles  

        self.possible_positions = [
            (x, y)
            for x in range(1, self.map_grid.shape[0] - 1)
            for y in range(1, self.map_grid.shape[1] - 1)
            if self.map_grid[x, y] == 1  
        ]
        
        self.position = random.choice(self.possible_positions)  

        self.target_position = self._set_initial_target()
        self.steps_tracker = {"moves": 0, "senses": 0, "detections": 0}
        self.recent_positions = deque(maxlen=self.oscillation_limit)
        self.path_to_rat = []
        self.path_logged = False

        self.pf = None

        self._determine_initial_location()  
        self.track_target()

    def _set_initial_target(self):
        return random.choice(self.possible_positions)

    def _check_surroundings(self):
        if self.position is None:
            raise ValueError("Bot position is not yet localized!")

        x, y = self.position
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        obstacles = sum(
            self.map_grid[x + dx, y + dy] == 0
            for dx, dy in directions
            if 0 <= x + dx < self.map_grid.shape[0] and 0 <= y + dy < self.map_grid.shape[1]
        )
        self.steps_tracker["senses"] += 1
        return obstacles

    def _determine_initial_location(self):
        print("Determining Bot Position...")
        possible_locations = set(self.possible_positions)
        attempt_limit = 100 
        attempts = 0

        while len(possible_locations) > 1 and attempts < attempt_limit:
            blocked_neighbors = self._check_surroundings()
            new_possible_locations = {
                (x, y) for (x, y) in possible_locations
                if self._count_obstacles(x, y) == blocked_neighbors
            }
            if not new_possible_locations:
                print("Warning: All possible locations eliminated. Retaining current possibilities.")
                break
            possible_locations = new_possible_locations
            if len(possible_locations) > 1:
                self.move_random_localization() 
                attempts += 1

        if possible_locations:
            self.position = next(iter(possible_locations))  
        else:
            raise ValueError("Failed to localize bot: No possible locations remain.")
        
        print("Localization phase complete.")
        print("Resolved Starting location:", self.position)

    def _count_obstacles(self, x, y):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return sum(
            self.map_grid[x + dx, y + dy] == 0
            for dx, dy in directions
            if 0 <= x + dx < self.map_grid.shape[0] and 0 <= y + dy < self.map_grid.shape[1]
        )


    def move_random_localization(self):
        x, y = self.position
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        possible_moves = [
            (dx, dy)
            for dx, dy in directions
            if 0 <= x + dx < self.map_grid.shape[0] and 0 <= y + dy < self.map_grid.shape[1]
            and self.map_grid[x + dx, y + dy] == 1 
        ]
        if possible_moves:
            direction = random.choice(possible_moves)
            self.position = (x + direction[0], y + direction[1])


    def move_towards_target(self):
        if self.path_to_rat:
            next_position = self.path_to_rat.pop(0)
            self.position = next_position
            self.steps_tracker["moves"] += 1
            self.recent_positions.append(self.position)

    def move_towards_most_probable_position(self):
        rat_estimate = self.pf.estimate_rat_position()

        rat_estimate = (int(rat_estimate[0]), int(rat_estimate[1]))
    
        print(f"Moving towards estimated rat location: {rat_estimate}")
        self.path_to_rat = self.find_shortest_path(self.position, rat_estimate)

        self.move_towards_target()

        if self.position == rat_estimate:
            print(f"Bot reached estimated rat position: {rat_estimate}. Checking if it matches actual rat position.")
            distance_to_rat = abs(self.position[0] - self.rat_position[0]) + abs(self.position[1] - self.rat_position[1])
            if distance_to_rat <= 1:  
                print("Bot has caught the rat!")
                return True  

        return False 


    def _sense_target(self):
            if self.position is None:
                raise ValueError("Bot position is not yet localized!")

            x_bot, y_bot = self.position
            x_rat, y_rat = self.target_position
            distance = abs(x_bot - x_rat) + abs(y_bot - y_rat)
            prob_ping = math.exp(-self.alpha * (distance - 1))
            self.steps_tracker["detections"] += 1
            return prob_ping >= self.threshold

    def update_target_knowledge(self, ping_result):
        if self.pf is None:  
            print("Initializing particles based on first ping...")
            self.pf = ParticleFilter(self.map_grid, num_particles=self.num_particles, alpha=self.alpha, threshold=self.threshold)
        
        self.pf.initialize_particles(rat_position=self.target_position) 
        self.pf.predict(self.position, ping_result) 
        self.pf.resample() 
  
    def break_oscillation(self):
        if len(self.recent_positions) == self.oscillation_limit:
            if len(set(self.recent_positions)) == 1:
                print("Oscillation detected. Moving randomly to escape.")
                self.move_random_localization()

    def find_shortest_path(self, start, goal):
        queue = deque([(start, [])])
        visited = set([start])
        while queue:
            current_pos, path = queue.popleft()
            if current_pos == goal:
                return path + [goal]

            x, y = current_pos
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_pos = (x + dx, y + dy)
                if (
                    0 <= next_pos[0] < self.map_grid.shape[0]
                    and 0 <= next_pos[1] < self.map_grid.shape[1]
                    and self.map_grid[next_pos[0], next_pos[1]] == 1  
                    and next_pos not in visited
                ):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [current_pos]))
        return []



    def track_target(self):
        running = True
        while running:

            if self.position == self.target_position:
                print("Bot has caught the rat!")
                # Print the final evaluation
                print("\nFinal Evaluation:")
                print(f"Total movements: {self.steps_tracker['moves']}")
                print(f"Total blocked senses: {self.steps_tracker['senses']}")
                print(f"Total rat detections: {self.steps_tracker['detections']}")
                print(f"Rat caught at: {self.position}")
                running = False

            self.break_oscillation()

            if not self.path_to_rat:
                ping_result = self._sense_target()
                if ping_result:
                    print(f"Ping detected at bot position {self.position}.")
                    self.path_to_rat = self.find_shortest_path(self.position, self.target_position)
                    if not self.path_logged:
                        print(f"Path to rat: {self.path_to_rat}")
                        self.path_logged = True
                else:
                    self.update_target_knowledge(ping_result)
                    self.move_towards_most_probable_position()
            else:
                self.move_towards_target()



if __name__ == "__main__":
    grid = create_ship_layout()
    bot = CustomBot(grid, alpha=0.1, threshold=0.7, decay=0.95, oscillation_limit=4)






