import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameters.
EXPLORATION_RATE = 0 # TODO fine tune this


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # TODO remove this model stuff
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
    # TODO Exploration vs exploitation
    if self.train and random.random() < EXPLORATION_RATE:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    # Load Q table.
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
    # TODO remove return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    # TODO I think we can just use a list and append without the need to stack
    channels = []

    pos = np.array(game_state.get("self")[3])    
    
    # Feature 1 - Dangerous zone: return a boolean that indicates wether the agent is in the path of an explosion
    danger = False
    timer = float('inf')

    for bomb in game_state.get("bombs"):
         if (bomb[0][0] == pos[0] or bomb[0][1] == pos[1]) and np.linalg.norm(np.array(bomb[0]) - pos) < 4:
            danger = True
            timer = bomb[1]
            break

    # Feature 2 & 3 - Nearest coin: return direction for nearest coin
    nearest_coin = [float('inf'), float('inf')]

    for coin in game_state.get("coins"):
        pos_coin = np.array(coin)
        if np.linalg.norm(pos_coin - pos) < np.linalg.norm(nearest_coin - pos):
            nearest_coin = pos_coin
    if nearest_coin[0] == float('inf'):
        coin_first_dir = ["FREE"]
    elif nearest_coin[0] - pos[0] > 0:
        coin_first_dir = ["RIGHT"]
    elif nearest_coin[0] - pos[0] < 0:
        coin_first_dir = ["LEFT"]  
    else:
        coin_first_dir = ["ALIGNED"]

    if nearest_coin[1] == float('inf'):
        coin_second_dir = ["FREE"]
    elif nearest_coin[1] - pos[1] < 0:
        coin_second_dir = ["DOWN"]
    elif nearest_coin[1] - pos[1] > 0:
        coin_second_dir = ["UP"]  
    else:
        coin_second_dir = ["ALIGNED"]

    #Feature 4 & 5 & 6 & 7 - Wall detection: returns -1 when walls and 0 when free tile
    vision_down = [game_state.get("field")[pos[0], pos[1] + 1]]
    vision_up = [game_state.get("field")[pos[0], pos[1] - 1]]
    vision_left = [game_state.get("field")[pos[0] - 1, pos[1]]]
    vision_right = [game_state.get("field")[pos[0] + 1, pos[1]]]

    #channels.append(danger) TODO insert later
    channels.append(coin_first_dir)
    channels.append(coin_second_dir)
    channels.append(vision_down)
    channels.append(vision_up)
    channels.append(vision_left)
    channels.append(vision_right)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels) #TODO remove
    # and return them as a vector
    return stacked_channels.reshape(-1)
    #return stacked_channels.reshape(-1)