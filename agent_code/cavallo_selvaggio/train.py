from collections import namedtuple, deque, defaultdict

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, convert

import os

import numpy as np
import pickle
import matplotlib.pyplot as plt #TODO add this as dependency

# Loading experience buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'weights'))

# Possible actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions TODO remove once sure not needed
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ... TODO remove once sure not needed
LEARNING_RATE = 0.05 # TODO fine tune this
DISCOUNT_RATE = 0.8 # TODO fine tune this
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


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.error = []

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
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

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
                                       

    # Debug messages for features
    #self.logger.debug("Closest coin is in " + self.transitions[-1].next_state[0] + " " + self.transitions[-1].next_state[1])
    self.logger.debug("Coin right? " + self.transitions[-1].next_state[0])  
    self.logger.debug("Coin left? " + self.transitions[-1].next_state[1])
    self.logger.debug("Coin up? " + self.transitions[-1].next_state[2])
    self.logger.debug("Coin down? " + self.transitions[-1].next_state[3])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - down " + self.transitions[-1].next_state[4])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - up " + self.transitions[-1].next_state[5])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - left " + self.transitions[-1].next_state[6])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - right " + self.transitions[-1].next_state[7])
    #self.logger.debug("Closest crate is in " + self.transitions[-1].next_state[6] + " " + self.transitions[-1].next_state[7])
    self.logger.debug("Crate right? " + self.transitions[-1].next_state[8])  
    self.logger.debug("Crate left? " + self.transitions[-1].next_state[9])
    self.logger.debug("Crate up? " + self.transitions[-1].next_state[10])
    self.logger.debug("Crate down? " + self.transitions[-1].next_state[11])
    self.logger.debug("Can he drop a bomb? " + self.transitions[-1].next_state[12])
    self.logger.debug("Can he destroy a crate from here? " + self.transitions[-1].next_state[13])
    self.logger.debug("Is he in danger? " + self.transitions[-1].next_state[14])
    self.logger.debug("Can he escape up? " + self.transitions[-1].next_state[15])
    self.logger.debug("Can he escape down? " + self.transitions[-1].next_state[16])
    self.logger.debug("Can he escape left? " + self.transitions[-1].next_state[17])
    self.logger.debug("Can he escape right? " + self.transitions[-1].next_state[18])
    # Debug message for events:
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    sum = 0
    n_a = 0
    n = min(TRANSITION_HISTORY_SIZE, len(self.transitions))
    for t in np.arange(-1, -n, -1):
        index = ACTIONS.index(self.transitions[t].action)
        if self.transitions[t].action == ACTIONS[index]:
            n_a += 1
            Y = self.transitions[t].reward + DISCOUNT_RATE * np.max(self.transitions[t].weights[index] @ convert(self,self.transitions[t].next_state))
            sum = sum + convert(self, self.transitions[t].state) * (Y - self.transitions[t].weights[index] @ convert(self,self.transitions[t].state))
        self.error.append((Y - self.weights[index] @ convert(self,self.transitions[t].state))**2)
        self.logger.debug("Error: " + str(self.error[-1]))
    index = ACTIONS.index(self_action)
    if n_a == 0:
        n_a = 1
    self.weights[index] = self.weights[index] + (LEARNING_RATE / n_a) * sum 


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    features = state_to_features(self, last_game_state)

    # Appending custom events to events:
    events = custom_events(self, last_action, features, [], events)

    # Appending transition:
    self.transitions.append(Transition(features, last_action, None, reward_from_events(self, events), self.weights))

    # Debug message for events:
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    sum = 0
    n = min(TRANSITION_HISTORY_SIZE, len(self.transitions))
    n_a = 0
    for t in np.arange(-2, -n, -1):
        index = ACTIONS.index(self.transitions[t].action)
        if self.transitions[t].action == ACTIONS[index]:
            n_a += 1
            Y = self.transitions[t].reward + DISCOUNT_RATE * np.max(self.transitions[t].weights[index] @ convert(self, self.transitions[t].next_state))
            sum = sum + convert(self, self.transitions[t].state) * (Y - (self.transitions[t].weights[index] @ convert(self, self.transitions[t].state)))
    index = ACTIONS.index(last_action)
    sum = convert(self, self.transitions[-1].state) * (self.transitions[-1].reward - self.weights[index] @ convert(self, self.transitions[-1].state))
    if n_a == 0:
        n_a = 1
    self.weights[index] = self.weights[index] + (LEARNING_RATE / n_a) * sum

    self.transitions.clear()
    # Store the weights
    with open("my-saved-weights.pt", "wb") as file:
        pickle.dump(self.weights, file)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        #COIN_NOT_COLLECTED: -.1,
        #GOING_AWAY_FROM_COIN: -.2,
        GOING_TO_COIN: 10,

        e.CRATE_DESTROYED: 10,
        BOMB_AND_CRATE: 5,
        GOING_TO_CRATE : 5,
        #GOING_AWAY_FROM_CRATE: -1,

        NO_ESCAPE: -100,
        e.GOT_KILLED: -1,
        e.KILLED_SELF: -50,
        #NO_ESCAPING: -3,
        ESCAPING: 50,
        #e.BOMB_DROPPED: -.1,
        NO_BOMB: -5,

        # e.KILLED_OPPONENT: 5,

        NOT_MOVING: -30,

        e.INVALID_ACTION: -1,
        TOO_WAITS: -10,
        GOING_INTO_WALL: -20,
        UNDECIDED: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def custom_events (self, self_action, old_features, new_features, events: List[str]) -> List[str]:
    # There's coin but it was not collected
    if not (e.COIN_COLLECTED in events) and not (str(old_features[0]) == "FREE" and str(old_features)[2] == "FREE" or str(old_features[1]) == "FREE" and str(old_features)[3] == "FREE"):
        events.append(COIN_NOT_COLLECTED)

    # Moving towards coin and going away from it
    if old_features[0] == self_action or old_features[1] == self_action or old_features[2] == self_action or old_features[3] == self_action:
        events.append(GOING_TO_COIN)
    elif not (old_features[0] == "FREE" and old_features[2] == "FREE" and old_features[1] == "FREE" and old_features[3] == "FREE"):
        events.append(GOING_AWAY_FROM_COIN)

    # Moving towards crate and going away from it
    if old_features[8] == self_action or old_features[9] == self_action or old_features[10] == self_action or old_features[11] == self_action:
        events.append(GOING_TO_CRATE)
    elif not (old_features[8] == "FREE" and old_features[10] == "FREE" and old_features[9] == "FREE" and old_features[11] == "FREE") and not self_action == "WAIT" and not self_action == "BOMB":
        events.append(GOING_AWAY_FROM_CRATE)

    if not len(new_features) == 0:
        if old_features[14] == -1:
            if old_features[15] == self_action or old_features[16] == self_action or old_features[17] == self_action or old_features[18] == self_action:
                events.append(ESCAPING)
            elif not(old_features[15] == 'NO DANGER AND CAN ESCAPE' and old_features[16] == 'NO DANGER AND CAN ESCAPE' and old_features[17] == 'NO DANGER AND CAN ESCAPE' and old_features[18] == 'NO DANGER AND CAN ESCAPE'):
                events.append(NO_ESCAPING)
        else:
            if self_action == "BOMB" and (old_features[15] == 'NO DANGER AND CAN ESCAPE' and old_features[16] == 'NO DANGER AND CAN ESCAPE' and old_features[17] == 'NO DANGER AND CAN ESCAPE' and old_features[18] == 'NO DANGER AND CAN ESCAPE'):
                events.append(NO_ESCAPE)
  

    # When he wants to hug walls (punish behaviour) TODO remove because invalid action
    features = tuple(old_features)
    if int(features[4]) == -1 and self_action == "DOWN":
        events.append(GOING_INTO_WALL)
    if int(features[5]) == -1 and self_action == "UP":
        events.append(GOING_INTO_WALL)
    if int(features[6]) == -1 and self_action == "LEFT":
        events.append(GOING_INTO_WALL)
    if int(features[7]) == -1 and self_action == "RIGHT":
        events.append(GOING_INTO_WALL)


    # Punish when he goes crazy
    if len(self.transitions) > 2:
        if self_action == self.transitions[-2].action and self.transitions[-1].action == self.transitions[-3].action:
            events.append(UNDECIDED)

    if len(self.transitions) > 4:
        if self_action == self.transitions[-3].action and self.transitions[-1].action == self.transitions[-4].action and self.transitions[-2].action == self.transitions[-5].action:
            events.append(UNDECIDED)

    if len(self.transitions) > 6:
        if self_action == self.transitions[-4].action and self.transitions[-1].action == self.transitions[-5].action and self.transitions[-2].action == self.transitions[-6].action and self.transitions[-3].action == self.transitions[-7].action:
            events.append(UNDECIDED)


    # When he puts a bomb close to a crate
    if int(features[13]) == 1 and self_action == "BOMB":
        events.append(BOMB_AND_CRATE)

    if (old_features[12] == -1) and (self_action == "BOMB"):
        events.append(NO_BOMB)

    if self_action == "WAIT" and not(old_features[4] == -1 and old_features[5] == -1 and old_features[6] == -1 and old_features[7] == -1):
        events.append(NOT_MOVING)


    return events