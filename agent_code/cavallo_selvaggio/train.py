from collections import namedtuple, deque, defaultdict

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, convert

import os

import numpy as np
import pickle

# Loading experience buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'weights'))

# Possible actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameters:
TRANSITION_HISTORY_SIZE = 50  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ... TODO remove once sure not needed
LEARNING_RATE = 0.03 
DISCOUNT_RATE = 0.8 
N_FEATURES = 19

# Custom events
COIN_NOT_COLLECTED = "COIN_NOT_COLLECTED"
GOING_TO_COIN = "GOING_TO_COIN"
GOING_TO_CRATE = "GOING_TO_CRATE"
GOING_INTO_WALL = "GOING_INTO_WALL"
UNDECIDED = "UNDECIDED"
BOMB_AND_CRATE = "BOMB_AND_CRATE"
TOO_WAITS = "TOO_WAITS"
NO_ESCAPE = "N0_ESCAPE"
ESCAPING = "ESCAPING"
GOING_AWAY_FROM_COIN = "GOING_AWAY_FROM_COIN"
GOING_AWAY_FROM_CRATE = "GOING_AWAY_FROM_CRATE"
NO_BOMB = "NO_BOMB"
NO_ESCAPING = "NO_ESCAPING"
NOT_MOVING = "NOT_MOVING"
USELESS = "USELESS"
OK_WAIT = "OK_WAIT"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Setup ransitions
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Load or create weights.
    if not os.path.isfile("my-saved-weights.pt"):
        self.weights = np.zeros((len(ACTIONS), N_FEATURES))
        with open("my-saved-weights.pt", "wb") as file:
            pickle.dump(self.weights, file)
    else:
        with open("my-saved-weights.pt", "rb") as file:
            self.weights = pickle.load(file)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events and to perform the update on weights.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    # Appending custom events to events:
    events = custom_events(self, self_action, old_features, new_features, events)

    # Appending transition:
    self.transitions.append(Transition(old_features, self_action, new_features, reward_from_events(self, events), self.weights))
                                       

    # Printing feature messages
    self.logger.info("Coin right? " + self.transitions[-1].next_state[0])  
    self.logger.info("Coin left? " + self.transitions[-1].next_state[1])
    self.logger.info("Coin up? " + self.transitions[-1].next_state[2])
    self.logger.info("Coin down? " + self.transitions[-1].next_state[3])
    self.logger.info("Vision is 1 for non walkable and 0 otherwise - down " + self.transitions[-1].next_state[4])
    self.logger.info("Vision is 1 for non walkable and 0 otherwise - up " + self.transitions[-1].next_state[5])
    self.logger.info("Vision is 1 for non walkable and 0 otherwise - left " + self.transitions[-1].next_state[6])
    self.logger.info("Vision is 1 for non walkable and 0 otherwise - right " + self.transitions[-1].next_state[7])
    self.logger.info("Crate right? " + self.transitions[-1].next_state[8])  
    self.logger.info("Crate left? " + self.transitions[-1].next_state[9])
    self.logger.info("Crate up? " + self.transitions[-1].next_state[10])
    self.logger.info("Crate down? " + self.transitions[-1].next_state[11])
    self.logger.info("Can he drop a bomb? " + self.transitions[-1].next_state[12])
    self.logger.info("Can he destroy a crate from here? " + self.transitions[-1].next_state[13])
    self.logger.info("Is he in danger? " + self.transitions[-1].next_state[14])
    self.logger.info("Can he escape up? " + self.transitions[-1].next_state[15])
    self.logger.info("Can he escape down? " + self.transitions[-1].next_state[16])
    self.logger.info("Can he escape left? " + self.transitions[-1].next_state[17])
    self.logger.info("Can he escape right? " + self.transitions[-1].next_state[18])

    # Info message for encountered events:
    self.logger.info(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Computing the update:
    sum = 0
    count = 0
    n = min(TRANSITION_HISTORY_SIZE, len(self.transitions))
    index = ACTIONS.index(self_action)
    # Cycling through last n transitions
    for t in np.arange(-1, -n - 1, -1):
        # Keeping only those with the same action as the last
        if self.transitions[t].action == self_action:
            count += 1
            Y = self.transitions[t].reward + DISCOUNT_RATE * np.max(self.transitions[t].weights[index] @ convert(self,self.transitions[t].next_state))
            sum = sum + convert(self, self.transitions[t].state) * (Y - self.transitions[t].weights[index] @ convert(self,self.transitions[t].state))
            
    index = ACTIONS.index(self_action)
    if count == 0:
        count = 1
    self.weights[index] = self.weights[index] + (LEARNING_RATE / count) * sum 


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    :param self: The same object that is passed to all of your callbacks.
    """

    global LEARNING_RATE
    global EXPLORATION_RATE
    LEARNING_RATE = LEARNING_RATE * 0.95

    if self.error:
        with open("error_log.txt", "a") as file_log:
            file_log.write(f"{LEARNING_RATE}, {self.error[-1]}\n")

    features = state_to_features(self, last_game_state)

    # Appending custom events to events:
    events = custom_events(self, last_action, features, [], events)

    # Appending transition:
    self.transitions.append(Transition(features, last_action, None, reward_from_events(self, events), self.weights))

    # Info message for events:
    self.logger.info(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    sum = 0
    n = min(TRANSITION_HISTORY_SIZE, len(self.transitions))
    count = 0
    index = ACTIONS.index(last_action)
    for t in np.arange(-2, -n - 1, -1):
        if self.transitions[t].action == last_action:
            count += 1
            Y = self.transitions[t].reward + DISCOUNT_RATE * np.max(self.transitions[t].weights[index] @ convert(self, self.transitions[t].next_state))
            sum = sum + convert(self, self.transitions[t].state) * (Y - (self.transitions[t].weights[index] @ convert(self, self.transitions[t].state)))
    # Last transition doesn't have prediction factor.
    sum = convert(self, self.transitions[-1].state) * (self.transitions[-1].reward - self.weights[index] @ convert(self, self.transitions[-1].state))
    if count == 0:
        count = 1
    self.weights[index] = self.weights[index] + (LEARNING_RATE / count) * sum

    # Clearing experience buffer:
    self.transitions.clear()

    # Store the weights
    with open("my-saved-weights.pt", "wb") as file:
        pickle.dump(self.weights, file)



def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent get so as to en/discourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,
        #COIN_NOT_COLLECTED: -.1,
        GOING_AWAY_FROM_COIN: -15,
        GOING_TO_COIN: 15,

        #e.CRATE_DESTROYED: 10,
        BOMB_AND_CRATE: 5,
        GOING_TO_CRATE : 2,
        GOING_AWAY_FROM_CRATE: -5,

        NO_ESCAPE: -25,
        #e.GOT_KILLED: -10,
        e.KILLED_SELF: -25,
        NO_ESCAPING: -30,
        ESCAPING: 30,
        #e.BOMB_DROPPED: -.1,
        NO_BOMB: -15,

        # e.KILLED_OPPONENT: 5,

        NOT_MOVING: -30,
        OK_WAIT: 1, 

        #e.INVALID_ACTION: -1,
        TOO_WAITS: -10,
        GOING_INTO_WALL: -15,
        UNDECIDED: -30,
        USELESS: -30,
    }
    # Compute reward sum:
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum}")
    return reward_sum



def custom_events (self, self_action, old_features, new_features, events: List[str]) -> List[str]:
    """
    Appending custom events to the list of official events.
    """

    # There's coin but it was not collected
    if not (e.COIN_COLLECTED in events) and not (str(old_features[0]) == "FREE" and str(old_features[2]) == "FREE" and str(old_features[1]) == "FREE" and str(old_features[3]) == "FREE"):
        events.append(COIN_NOT_COLLECTED)

    # Moving towards coin and going away from it
    if (str(old_features[0]) == self_action) or (str(old_features[1]) == self_action) or (str(old_features[2]) == self_action) or (str(old_features[3]) == self_action):
        events.append(GOING_TO_COIN)
    elif not ((str(old_features[0]) == "FREE") and (str(old_features[2]) == "FREE") and (str(old_features[1]) == "FREE") and (str(old_features[3]) == "FREE")):
        events.append(GOING_AWAY_FROM_COIN)

    # Moving towards crate and going away from it
    if (str(old_features[8]) == self_action) or (str(old_features[9]) == self_action) or (str(old_features[10]) == self_action) or (str(old_features[11]) == self_action):
        events.append(GOING_TO_CRATE)
    elif (not ((str(old_features[8]) == "FREE") and (str(old_features[10]) == "FREE") and (str(old_features[9]) == "FREE") and (str(old_features[11]) == "FREE"))) and (not self_action == "WAIT") and (not self_action == "BOMB"):
        events.append(GOING_AWAY_FROM_CRATE)

    # Escaping from danger, not escaping from danger and dropping a bomb by trapping himself.
    if int(old_features[14]) == 1:
        if str(old_features[15]) == self_action or str(old_features[16]) == self_action or str(old_features[17]) == self_action or str(old_features[18]) == self_action:
            events.append(ESCAPING)
        elif not(str(old_features[15]) == 'NO ESCAPE' and str(old_features[16]) == 'NO ESCAPE' and str(old_features[17]) == 'NO ESCAPE' and str(old_features[18]) == 'NO ESCAPE'):
            events.append(NO_ESCAPING)
    else:
        if self_action == "BOMB" and (str(old_features[15]) == 'NO ESCAPE' and str(old_features[16]) == 'NO ESCAPE' and str(old_features[17]) == 'NO ESCAPE' and str(old_features[18]) == 'NO ESCAPE'):
            events.append(NO_ESCAPE)
  

    # Punish when he wants to walk towards forbidden areas.
    features = tuple(old_features)
    if int(old_features[4]) == 1 and self_action == "DOWN":
        events.append(GOING_INTO_WALL)
    if int(old_features[5]) == 1 and self_action == "UP":
        events.append(GOING_INTO_WALL)
    if int(old_features[6]) == 1 and self_action == "LEFT":
        events.append(GOING_INTO_WALL)
    if int(old_features[7]) == 1 and self_action == "RIGHT":
        events.append(GOING_INTO_WALL)


    # Punish when he repeats the same move for too long
    if len(self.transitions) > 2:
        if self_action == self.transitions[-2].action and self.transitions[-1].action == self.transitions[-3].action:
            events.append(UNDECIDED)

    if len(self.transitions) > 4:
        if self_action == self.transitions[-3].action and self.transitions[-1].action == self.transitions[-4].action and self.transitions[-2].action == self.transitions[-5].action:
            events.append(UNDECIDED)

    if len(self.transitions) > 6:
        if self_action == self.transitions[-4].action and self.transitions[-1].action == self.transitions[-5].action and self.transitions[-2].action == self.transitions[-6].action and self.transitions[-3].action == self.transitions[-7].action:
            events.append(UNDECIDED)


    # When he puts a bomb cin a place where he'd destroy a crate
    if (int(old_features[13]) == 1) and (self_action == "BOMB") and (int(old_features[12]) == 1) and not ((str(old_features[15]) == 'NO ESCAPE' and str(old_features[16]) == 'NO ESCAPE' and str(old_features[17]) == 'NO ESCAPE' and str(old_features[18]) == 'NO ESCAPE')):
        events.append(BOMB_AND_CRATE)

    # When he wants to drop bombs while he can't.
    if (int(old_features[12]) == 0) and (self_action == "BOMB"):
        events.append(NO_BOMB)

    # When he waits while there's no point in doing so.
    if (self_action == "WAIT") and not(int(old_features[4]) == 1 and int(old_features[5]) == 1 and int(old_features[6]) == 1 and int(old_features[7]) == 1):
        events.append(NOT_MOVING)
    
    # When he drops a bomb that wouldn't destroy a crate
    if (self_action == "BOMB") and (int(old_features[13]) == 0):
        events.append(USELESS)
    
    # When he waits for a good reason.
    if self_action == "WAIT" and (int(old_features[4]) == 1 and int(old_features[5]) == 1 and int(old_features[6]) == 1 and int(old_features[7]) == 1):
        events.append(OK_WAIT)

    return events