import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameters.
EXPLORATION_RATE = 0.9 # TODO fine tune this


def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # TODO remove this model stuff when sure it's not needed
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if self.train and random.random() < EXPLORATION_RATE:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    # Load or create Q table.
    if not os.path.isfile("q-table.pt"):
        self.q_table = defaultdict(int)
        with open("q-table.pt", "wb") as file:
            pickle.dump(self.q_table, file)
    elif not self.train:
        with open("q-table.pt", "rb") as file:
            self.q_table = pickle.load(file)
    
    # Find action maximizing Q value.
    q_value = float('-inf')
    best_action = 'WAIT'

    for action in ACTIONS:
        if self.q_table[tuple(state_to_features(game_state)), action] > q_value:
            q_value = self.q_table[(tuple(state_to_features(game_state)), action)]
            best_action = action

    self.logger.debug(best_action)

    return best_action


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # TODO I think we can just use a list and append without the need to stack
    channels = []

    current_position = np.array(game_state.get("self")[3])    
    
    # Feature 1 - Dangerous zone: return a boolean that indicates wether the agent is in the path of an explosion
    danger = False
    timer = float('inf')

    for bomb in game_state.get("bombs"):
         if (bomb[0][0] == current_position[0] or bomb[0][1] == current_position[1]) and np.linalg.norm(np.array(bomb[0]) - current_position) < 4:
            danger = True
            timer = bomb[1] #TODO remove maybe
            break

    # Feature 2 & 3 - Nearest coin: return direction for nearest coin
    nearest_coin = [float('inf'), float('inf')]
    coin_first_dir = ["FREE"]
    coin_second_dir = ["FREE"]
    map_size = game_state.get("field").shape[0]
    flag = 0

    for coin in game_state.get("coins"):
        pos_coin = np.array(coin)
        if np.linalg.norm(pos_coin - current_position) < np.linalg.norm(nearest_coin - current_position):
            nearest_coin = pos_coin
    if nearest_coin[0] == float('inf'):
        for raggio in range(1, map_size + 1):
                for i in range(current_position[0] - raggio, current_position[0] + raggio + 1):
                    if i > 0 or i < map_size:
                        for j in range(current_position[1] - raggio, current_position[1] + raggio + 1):
                            if j > 0 or j < map_size:
                                if (i, j) != (current_position[0], current_position[1]) and abs(current_position[0] - i) == raggio or abs(current_position[1] - j) == raggio:
                                    if game_state.get("field")[i,j] == 1:
                                        flag = 1
                                        if i - current_position[0] > 0:
                                            coin_first_dir = ["RIGHT"]
                                        elif i - current_position[0] < 0:
                                            coin_first_dir = ["LEFT"]  
                                        else:
                                            coin_first_dir = ["ALIGNED"]
                                        if j - current_position[1] < 0:
                                            coin_second_dir = ["DOWN"]
                                        elif j - current_position[1] > 0:
                                            coin_second_dir = ["UP"]  
                                        else:
                                            coin_second_dir = ["ALIGNED"]
                                        break
                        if flag:
                            break
                if flag:
                    break   
    elif nearest_coin[0] - current_position[0] > 0:
        coin_first_dir = ["RIGHT"]
    elif nearest_coin[0] - current_position[0] < 0:
        coin_first_dir = ["LEFT"]  
    else:
        coin_first_dir = ["ALIGNED"]

    if nearest_coin[1] - current_position[1] < 0:
        coin_second_dir = ["DOWN"]
    elif nearest_coin[1] - current_position[1] > 0:
        coin_second_dir = ["UP"]  
    else:
        coin_second_dir = ["ALIGNED"]

    #Feature 4 & 5 & 6 & 7 - Wall detection: returns -1 when non-walkable tile and 0 when free tile
    vision_down = [1 if game_state.get("field")[current_position[0], current_position[1] + 1]  or (any(x == [current_position[0], current_position[1] + 1] for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 0]
    vision_up = [1 if game_state.get("field")[current_position[0], current_position[1] - 1] or (any(x == [current_position[0], current_position[1] - 1] for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 0]
    vision_left = [1 if game_state.get("field")[current_position[0] - 1, current_position[1]] or (any(x == [current_position[0] - 1, current_position[1]] for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 0]
    vision_right = [1 if game_state.get("field")[current_position[0] + 1, current_position[1]] or (any(x == [current_position[0] + 1, current_position[1]] for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 0]

    vision_crate = 0
    for i in range(2):
        for j in range(2):
            if game_state.get("field")[current_position[0] + i % 2, current_position[1] + j % 2] == 1:
                vision_crate = 1
                break
        if game_state.get("field")[current_position[0] + i % 2, current_position[1] + j % 2] == 1:
            break




    # Computing coordinates where there will be an explosion
    blast_coords = []
    for bomb in game_state.get("bombs"):
        x, y = bomb[0][0], bomb[0][1]
        blast_coords = [(x, y)]
        for i in range(1, 4):
            if game_state.get("field")[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, 4):
            if game_state.get("field")[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, 4):
            if game_state.get("field")[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, 4):
            if game_state.get("field")[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

    # Feature 8 & 9 & 10 & 11 - Danger detection: returns 1 when in that direction there will be an explosion, -1 when in that direction there currently is an explosion and 0 otherwise
    danger_down = [1 if ((current_position[0], current_position[1] + 1) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0], current_position[1] + 1] > 0 else 0]
    danger_up = [1 if ((current_position[0], current_position[1] - 1) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0], current_position[1] - 1] > 0 else 0]
    danger_left = [1 if ((current_position[0] - 1, current_position[1]) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0] - 1, current_position[1]] > 0 else 0]
    danger_right = [1 if ((current_position[0] + 1, current_position[1]) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0] + 1, current_position[1]] > 0 else 0]

    # Appending every feature
    channels.append([danger])
    channels.append(coin_first_dir)
    channels.append(coin_second_dir)
    channels.append(vision_down)
    channels.append(vision_up)
    channels.append(vision_left)
    channels.append(vision_right)
    channels.append(danger_down)
    channels.append(danger_up)
    channels.append(danger_left)
    channels.append(danger_right)
    channels.append([vision_crate])

    # Concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # ... and return them as a vector
    return stacked_channels.reshape(-1)

#TODO Mettere vision crate in features; Dirgli se va a sbattere contro cesta (punirlo); 