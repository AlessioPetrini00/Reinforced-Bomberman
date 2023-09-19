import os
import pickle
import random
import numpy as np
from collections import defaultdict


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameters.
EXPLORATION_RATE = 1 # TODO fine tune this

def setup(self):
    """
    Setup your code. This is called once when loading each agent.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Load or create Q table.
    if not os.path.isfile("q-table.pt"):
        self.q_table = defaultdict(int)
        with open("q-table.pt", "wb") as file:
            pickle.dump(self.q_table, file)
    else:
        with open("q-table.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    #Exploration path
    if self.train and random.random() < EXPLORATION_RATE:
        # 80%: walk in any direction. 10% wait. 10% bomb.
        choice = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

        self.logger.debug("Returning random choice: " + choice)

        return choice
    
    # Exploitation path
    # Find action maximizing Q value.
    q_value = float('-inf')
    best_action = 'WAIT'

    for action in ACTIONS:
        features = tuple(state_to_features(game_state))
        self.logger.debug("The value for action " + action + " is " + str(self.q_table[(features, action)]))
        if self.q_table[features, action] > q_value:
            q_value = self.q_table[(features, action)]
            best_action = action

    self.logger.debug("La migliore azione è " + best_action)

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

    # VARIABLES:
    channels = []
    current_position = np.array(game_state.get("self")[3])    
    
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
    
    map_size = game_state.get("field").shape[0]

    # FEATURES:
    # Feature 1 - Is the agent in danger
    danger = [1 if ((current_position[0], current_position[1]) in blast_coords if blast_coords else False) else 0]
    
    # Feature 2 & 3 - Nearest coin: return direction for nearest coin or "FREE" "FREE" if there's no coin
    nearest_coin = [float('inf'), float('inf')]
    coin_first_dir = ["FREE"]
    coin_second_dir = ["FREE"]

    # Find coordinates for nearest coin
    for coin in game_state.get("coins"):
        pos_coin = np.array(coin)
        if np.linalg.norm(pos_coin - current_position) < np.linalg.norm(nearest_coin - current_position):
            nearest_coin = pos_coin
    
    # When no coins look for nearest crate
    if not nearest_coin[0] == float('inf'):
        # Compute direction from coin position
        if nearest_coin[0] - current_position[0] > 0:
            coin_first_dir = ["RIGHT"]
        elif nearest_coin[0] - current_position[0] < 0:
            coin_first_dir = ["LEFT"]  
        else:
            coin_first_dir = ["ALIGNED"]

    if not nearest_coin[1] == float('inf'):
        if nearest_coin[1] - current_position[1] < 0:
            coin_second_dir = ["UP"]
        elif nearest_coin[1] - current_position[1] > 0:
            coin_second_dir = ["DOWN"]  
        else:
            coin_second_dir = ["ALIGNED"]

    #Feature 4 & 5 & 6 & 7 - Obstacle detection: returns 1 when non-walkable tile, 2 if crate and 0 when free tile TODO detect other players
    vision_down = [1 if game_state.get("field")[current_position[0], current_position[1] + 1] == -1  or (any(x == (current_position[0], current_position[1] + 1) for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 2 if game_state.get("field")[current_position[0], current_position[1] + 1] == 1 else 0]
    vision_up = [1 if game_state.get("field")[current_position[0], current_position[1] - 1] == -1 or (any(x == (current_position[0], current_position[1] - 1) for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 2 if game_state.get("field")[current_position[0], current_position[1] - 1] == 1 else 0]
    vision_left = [1 if game_state.get("field")[current_position[0] - 1, current_position[1]] == -1 or (any(x == (current_position[0] - 1, current_position[1]) for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 2 if game_state.get("field")[current_position[0] - 1, current_position[1]] == 1 else 0]
    vision_right = [1 if game_state.get("field")[current_position[0] + 1, current_position[1]] == -1 or (any(x == (current_position[0] + 1, current_position[1]) for x, _ in game_state.get("bombs")) if game_state.get("bombs") else False) else 2 if game_state.get("field")[current_position[0] + 1, current_position[1]] == 1 else 0]


    # Feature 8 & 9 & 10 & 11 - Danger detection: returns 1 when in that direction there will be an explosion, -1 when in that direction there currently is an explosion and 0 otherwise
    danger_down = [1 if ((current_position[0], current_position[1] + 1) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0], current_position[1] + 1] > 0 else 0]
    danger_up = [1 if ((current_position[0], current_position[1] - 1) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0], current_position[1] - 1] > 0 else 0]
    danger_left = [1 if ((current_position[0] - 1, current_position[1]) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0] - 1, current_position[1]] > 0 else 0]
    danger_right = [1 if ((current_position[0] + 1, current_position[1]) in blast_coords if blast_coords else False) else -1 if game_state.get("explosion_map")[current_position[0] + 1, current_position[1]] > 0 else 0]


    # Feature 12 - Crate vision: returns 1 if he's close to crate 0 otherwise
    # vision_crate = 0
    # for i,j in [(-1,0), (1, 0), (0,1), (0, -1)]:
    #         if game_state.get("field")[current_position[0] + i, current_position[1] + j] == 1:
    #             vision_crate = 1
    #             break

    # Feature 13 & 14 & 15 & 16 - Escape possibility: returns 1 if in that direction there's a possible escape in the hypothesis that he drops a bomb now
    escape_up = 0
    for i in np.arange(1,4):
        if game_state.get("field")[current_position[0], current_position[1] - i] == 1 or game_state.get("field")[current_position[0], current_position[1] - i] == -1:
            break
        elif game_state.get("field")[current_position[0] + 1, current_position[1] - i] == 0 or game_state.get("field")[current_position[0] - 1, current_position[1] - i] == 0:
            escape_up = 1
            break
        elif i == 3 and game_state.get("field")[current_position[0], current_position[1] - i - 1] == 0:
            escape_up = 1
            break
    escape_down = 0
    for i in np.arange(1,4):
        if game_state.get("field")[current_position[0], current_position[1] + i] == 1 or game_state.get("field")[current_position[0], current_position[1] + i] == -1:
            break
        elif game_state.get("field")[current_position[0] + 1, current_position[1] + i] == 0 or game_state.get("field")[current_position[0] - 1, current_position[1] + i] == 0:
            escape_down = 1
            break
        elif i == 3 and game_state.get("field")[current_position[0], current_position[1] + i + 1] == 0:
            escape_down = 1
            break
    escape_left = 0
    for i in np.arange(1,4):
        if game_state.get("field")[current_position[0] - i, current_position[1]] == 1 or game_state.get("field")[current_position[0] - i, current_position[1]] == -1:
            break
        elif game_state.get("field")[current_position[0] - i, current_position[1] - 1] == 0 or game_state.get("field")[current_position[0] - i, current_position[1] + 1] == 0:
            escape_left = 1
            break
        elif i == 3 and game_state.get("field")[current_position[0] - i - 1, current_position[1]] == 0:
            escape_left = 1
            break
    escape_right = 0
    for i in np.arange(1,4):
        if game_state.get("field")[current_position[0] + i, current_position[1]] == 1 or game_state.get("field")[current_position[0] + i, current_position[1]] == -1:
            break
        elif game_state.get("field")[current_position[0] + i, current_position[1] - 1] == 0 or game_state.get("field")[current_position[0] + i, current_position[1] + 1] == 0:
            escape_right = 1
            break
        elif i == 3 and game_state.get("field")[current_position[0] + i + 1, current_position[1]] == 0:
            escape_right = 1
            break

    #
    crate_first_dir = ["FREE"]
    crate_second_dir = ["FREE"]
    flag = 0
    for raggio in range(1, map_size + 1):
            for i in range(current_position[0] - raggio, current_position[0] + raggio + 1):
                if i > 0 and i < map_size:
                    for j in range(current_position[1] - raggio, current_position[1] + raggio + 1):
                        if j > 0 and j < map_size:
                            if (i, j) != (current_position[0], current_position[1]) and (abs(current_position[0] - i) == raggio or abs(current_position[1] - j) == raggio):
                                if game_state.get("field")[i,j] == 1:
                                    flag = 1
                                    if i - current_position[0] > 0:
                                        crate_first_dir = ["RIGHT"]
                                    elif i - current_position[0] < 0:
                                        crate_first_dir = ["LEFT"]  
                                    else:
                                        crate_first_dir = ["ALIGNED"]
                                    if j - current_position[1] < 0:
                                        crate_second_dir = ["UP"]
                                    elif j - current_position[1] > 0:
                                        crate_second_dir = ["DOWN"]  
                                    else:
                                        crate_second_dir = ["ALIGNED"]
                                    break
                    if flag:
                        break
            if flag:
                break

    #Feature 19 - Destroyable crate: returns 1 if by dropping a bomb in agent position he would destroy a crate
    destroyable_crates = 0
    x, y = current_position[0], current_position[1]
    blast_coords = [(x, y)]
    for i in range(1, 4):
        if game_state.get("field")[x + i, y] == -1:
            break
        if game_state.get("field")[x + i, y] == 1:
            destroyable_crates = 1
    if not destroyable_crates:
        for i in range(1, 4):
            if game_state.get("field")[x - i, y] == -1:
                break
            if game_state.get("field")[x - i, y] == 1:
                destroyable_crates = 1
    if not destroyable_crates:
        for i in range(1, 4):
            if game_state.get("field")[x, y + i] == -1:
                break
            if game_state.get("field")[x, y + i] == 1:
                destroyable_crates = 1
    if not destroyable_crates:
        for i in range(1, 4):
            if game_state.get("field")[x, y - i] == -1:
                break
            if game_state.get("field")[x, y - i] == 1:
                destroyable_crates = 1


    # Appending every feature
    channels.append(danger) #0
    channels.append(coin_first_dir) #1
    channels.append(coin_second_dir) #2
    channels.append(vision_down) #3
    channels.append(vision_up) #4
    channels.append(vision_left) #5
    channels.append(vision_right) #6
    channels.append(danger_down) #7
    channels.append(danger_up) #8
    channels.append(danger_left) #9
    channels.append(danger_right) #10
    channels.append([escape_up]) #11
    channels.append([escape_down]) #12
    channels.append([escape_left]) #13
    channels.append([escape_right]) #14
    channels.append(crate_first_dir) #15
    channels.append(crate_second_dir) #16
    channels.append([game_state.get("self")[2]]) #17 - Can he drop bomb?
    channels.append([destroyable_crates]) #18


    # Concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # ... and return them as a vector
    return stacked_channels.reshape(-1)


#TODO blast coords in escape