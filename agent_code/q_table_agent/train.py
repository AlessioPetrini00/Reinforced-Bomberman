from collections import namedtuple, deque, defaultdict

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import os

import numpy as np

from .callbacks import EXPLORATION_RATE

# Loading experience buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Possible actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 8  # keep only ... last transitions 
LEARNING_RATE = 0.9 # TODO fine tune this
DISCOUNT_RATE = 0.1 # TODO fine tune this

# Custom events
COIN_NOT_COLLECTED = "COIN_NOT_COLLECTED"
GOING_TO_COIN = "GOING_TO_COIN"
GOING_TO_CRATE = "GOING_TO_CRATE"
GOING_AWAY_FROM_BOMB = "GOING_AWAY_FROM_BOMB"
GOING_INTO_WALL = "GOING_INTO_WALL"
UNDECIDED = "UNDECIDED"
GOING_TO_BOMB = "GOING_TO_BOMB"
BOMB_AND_CRATE = "BOMB_AND_CRATE"
TOO_WAITS = "TOO_WAITS"
NO_ESCAPE = "N0_ESCAPE"
ESCAPING = "ESCAPING"
GOING_AWAY_FROM_COIN = "GOING_AWAY_FROM_COIN"
GOING_AWAY_FROM_CRATE = "GOING_AWAY_FROM_CRATE"
NO_BOMB = "NO_BOMB"



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Load or create Q table.
    if not os.path.isfile("q-table.pt"):
        self.q_table = defaultdict(int)
        with open("q-table.pt", "wb") as file:
            pickle.dump(self.q_table, file)
    else:
        with open("q-table.pt", "rb") as file:
            self.q_table = pickle.load(file)


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
    self.transitions.append(Transition(old_features, self_action, new_features, reward_from_events(self, events)))

    # Debug messages for features
    self.logger.debug("Closest coin is in " + self.transitions[-1].next_state[0] + " " + self.transitions[-1].next_state[1])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - down " + self.transitions[-1].next_state[2])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - up " + self.transitions[-1].next_state[3])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - left " + self.transitions[-1].next_state[4])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - right " + self.transitions[-1].next_state[5])
    self.logger.debug("Closest crate is in " + self.transitions[-1].next_state[6] + " " + self.transitions[-1].next_state[7])
    self.logger.debug("Can he drop a bomb? " + self.transitions[-1].next_state[8])
    self.logger.debug("Can he destroy a crate from here? " + self.transitions[-1].next_state[9])
    self.logger.debug("Danger info: " + self.transitions[-1].next_state[10])

    # Debug message for events:
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Perform update of the Q table.
    # We don't store in q_table.pt because we found out it takes a lot of computational time
    self.q_table[tuple(self.transitions[-1].state),self_action] = (1 - LEARNING_RATE) * self.q_table[tuple(self.transitions[-1].state),self_action] + (LEARNING_RATE) * (self.transitions[-1].reward + (DISCOUNT_RATE) * value_function(self, tuple(self.transitions[-1].next_state)))
    for action in ACTIONS:
        self.logger.debug("The value for action " + action + " is " + str(self.q_table[(tuple(self.transitions[-1].state), action)]))



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
    self.transitions.append(Transition(features, last_action, None, reward_from_events(self, events)))

    # Debug message for events:
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Update the Q table.
    self.q_table[tuple(self.transitions[-1].state),last_action] = (1 - (LEARNING_RATE)) * self.q_table[tuple(self.transitions[-1].state),last_action] + (LEARNING_RATE) * self.transitions[-1].reward

    # Store the new Q table so that it can be used in the next game.
    with open("q-table.pt", "wb") as file:
        pickle.dump(self.q_table, file)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        COIN_NOT_COLLECTED: -2,
        GOING_AWAY_FROM_COIN: -3,
        GOING_TO_COIN: 5,

        e.CRATE_DESTROYED: 1,
        BOMB_AND_CRATE: 2,
        GOING_TO_CRATE : 1,
        GOING_AWAY_FROM_CRATE: -1,

        #GOING_AWAY_FROM_BOMB: 8,
        NO_ESCAPE: -80,
        #GOING_TO_BOMB: -9,
        e.GOT_KILLED: -0.5,
        e.KILLED_SELF: -30,

        # e.KILLED_OPPONENT: 5,
        e.BOMB_DROPPED: -1,
        e.INVALID_ACTION: -50,
        # e.WAITED:,
        #TOO_WAITS: -3,
        GOING_INTO_WALL: -20,
        UNDECIDED: -6,
        NO_BOMB: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def value_function(self, features)->float:
    # The value function returns the maximum value of the Q table for a given state, iterating through all the possible actions.
    value = float('-inf')

    for action in ACTIONS:
        if self.q_table[features, action] > value:
            value = self.q_table[features, action]

    return value

def custom_events (self, self_action, old_features, new_features, events: List[str]) -> List[str]:
    # There's coin but it was not collected
    if not (e.COIN_COLLECTED in events) and not str(old_features[0]) == "FREE":
        events.append(COIN_NOT_COLLECTED)

    # Moving towards coin and going away from it
    if old_features[0] == self_action or old_features[1] == self_action:
        events.append(GOING_TO_COIN)
    elif not (old_features[0] == "FREE" and old_features[1] == "FREE"):
        events.append(GOING_AWAY_FROM_COIN)

    # Moving towards crate and going away from it
    if old_features[6] == self_action or old_features[7] == self_action:
        events.append(GOING_TO_CRATE)
    elif not (old_features[6] == "FREE" and old_features[7] == "FREE") and not self_action == "WAIT" and not self_action == "BOMB":
        events.append(GOING_AWAY_FROM_CRATE)
    
    # Remaining or going to dangerous-zone
    if not len(new_features) == 0:
        if (str(old_features[10]) == "NO DANGER AND CAN ESCAPE" or str(old_features[10]) == "NO DANGER NO ESCAPE") and not (str(new_features[10]) == "NO DANGER AND CAN ESCAPE" or str(new_features[10]) == "NO DANGER NO ESCAPE") and not (self_action == "BOMB" or self_action == "WAIT"):
            events.append(GOING_TO_BOMB)

    # NO_ESCAPE when the agent traps himself and ESCAPING if he picks a correct escape direction
    # if not len(new_features) == 0:
    #     if str(new_features[10]) == "NO ESCAPE":
    #         events.append(NO_ESCAPE)
    if self_action == "BOMB" and str(old_features[10]) == "NO DANGER NO ESCAPE":
        events.append(NO_ESCAPE)
    if self_action == str(old_features[10]):
        events.append(ESCAPING)


    # When he wants to hug walls (punish behaviour) TODO remove because invalid action
    features = tuple(old_features)
    if int(features[2]) == 1 and self_action == "DOWN":
        events.append(GOING_INTO_WALL)
    if int(features[3]) == 1 and self_action == "UP":
        events.append(GOING_INTO_WALL)
    if int(features[4]) == 1 and self_action == "LEFT":
        events.append(GOING_INTO_WALL)
    if int(features[5]) == 1 and self_action == "RIGHT":
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
    if int(features[9]) == 1 and self_action == "BOMB":
        events.append(BOMB_AND_CRATE)

    if (old_features[8] == 0) and (self_action == "BOMB"):
        events.append(NO_BOMB)
    
    return events