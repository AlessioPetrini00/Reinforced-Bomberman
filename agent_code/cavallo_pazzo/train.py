from collections import namedtuple, deque, defaultdict

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import os

import numpy as np

import matplotlib.pyplot as plt #TODO add this as dependency

# Loading experience buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Possible actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions TODO remove once sure not needed
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ... TODO remove once sure not needed
LEARNING_RATE = 0.2 # TODO fine tune this
DISCOUNT_RATE = 0.8 # TODO fine tune this

# Custom events
COIN_NOT_COLLECTED = "COIN_NOT_COLLECTED"
BOMB_MISSED = "BOMB_MISSED"
GOING_TO_COIN_OR_CRATE = "GOING_TO_COIN_OR_CRATE"
GOING_AWAY_FROM_BOMB = "GOING_AWAY_FROM_BOMB"
GOING_INTO_WALL = "GOING_INTO_WALL"
UNDECIDED = "UNDECIDED"
GOING_TO_BOMB = "GOING_TO_BOMB"
BOMB_AND_CRATE = "BOMB_AND_CRATE"





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
    self.logger.debug("Closest coin or crate are in " + state_to_features(new_game_state)[1] + " " + state_to_features(new_game_state)[2])
    self.logger.debug("1 for danger, -1 explosion, 0 nothing. You're now in: "+state_to_features(new_game_state)[0])
    self.logger.debug("down "+state_to_features(new_game_state)[7])
    self.logger.debug("up "+state_to_features(new_game_state)[8])
    self.logger.debug("left "+state_to_features(new_game_state)[9])
    self.logger.debug("right "+state_to_features(new_game_state)[10])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - down "+state_to_features(new_game_state)[3])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - up "+state_to_features(new_game_state)[4])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - left "+state_to_features(new_game_state)[5])
    self.logger.debug("vision is 1 for non walkable and 0 otherwise - right "+state_to_features(new_game_state)[6])
    self.logger.debug("sees crate? "+state_to_features(new_game_state)[11])



    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # Custom event: coin not collected
    if not (new_game_state.get("self")[3] in old_game_state.get("coins")):
        events.append(COIN_NOT_COLLECTED)

    pos = np.array(new_game_state.get("self")[3])
    
    # Custom event: you moved towards the closest coin
    if old_game_state.get("coins"):
        nearest_coin = old_game_state.get("coins")[0]
        for coin in old_game_state.get("coins"):
            if np.linalg.norm(np.array(coin) - pos) < np.linalg.norm(np.array(nearest_coin) - pos):
                nearest_coin = coin
        old_pos = np.array(old_game_state.get("self")[3])
        if np.linalg.norm(nearest_coin - pos) < np.linalg.norm(nearest_coin - old_pos):
            events.append(GOING_TO_COIN_OR_CRATE)
    
    
    if self.transitions[-1].state[0] and not self.transitions[-1].next_state[0]:
        events.append(GOING_AWAY_FROM_BOMB)

    if not self.transitions[-1].state[0] and self.transitions[-1].next_state[0]:
        events.append(GOING_TO_BOMB)


    # Custom event: when he wants to hug walls (punish behaviour)
    features = state_to_features(old_game_state)
    if features[2] == -1 and self_action == "DOWN":
        events.append(GOING_INTO_WALL)
    if features[3] == -1 and self_action == "UP":
        events.append(GOING_INTO_WALL)
    if features[4] == -1 and self_action == "LEFT":
        events.append(GOING_INTO_WALL)
    if features[5] == -1 and self_action == "RIGHT":
        events.append(GOING_INTO_WALL)

    # Custom event: punish when he goes crazy
    if len(self.transitions) > 3:
        #print(self.transitions[-1].action, self.transitions[-2].action, self.transitions[-3].action, self.transitions[-4].action)
        if self.transitions[-1].action == self.transitions[-3].action and self.transitions[-2].action == self.transitions[-4].action:
            events.append(UNDECIDED)

    if features[11] == 1 and self_action == "BOMB":
        events.append(BOMB_AND_CRATE)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    # Perform update of the Q table.
    # We don't store in q_table.pt because we found out it takes a lot of computational time

    self.q_table[tuple(state_to_features(old_game_state)),self_action] = (1 - LEARNING_RATE) * self.q_table[tuple(state_to_features(old_game_state)),self_action] + LEARNING_RATE * ((reward_from_events(self, events)) + DISCOUNT_RATE * value_function(self, new_game_state))


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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # FIXME Perform update of the Q table - last_game_state is y not x!!!
    self.q_table[tuple(state_to_features(last_game_state)),last_action] = (1 - LEARNING_RATE) * self.q_table[tuple(state_to_features(last_game_state)),last_action] + LEARNING_RATE * (reward_from_events(self, events))

    # Store the new Q table so that it can be used in the next game.
    with open("q-table.pt", "wb") as file:
        pickle.dump(self.q_table, file)

    if (last_game_state.get("self")[0] == 1):
        print(self.q_table)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        COIN_NOT_COLLECTED: -1,
        # e.KILLED_OPPONENT: 5,
        e.BOMB_DROPPED: 4,
        e.INVALID_ACTION: -200,
        e.WAITED: -40,
        # e.GOT_KILLED: -100,
        e.KILLED_SELF: -50,
        e.CRATE_DESTROYED: 5,

        # BOMB_AND_CRATE: 5,
        GOING_AWAY_FROM_BOMB: 45, 
        GOING_INTO_WALL: -200,
        GOING_TO_COIN_OR_CRATE: 4,
        #COIN_NOT_COLLECTED: -30,
        # GOING_TO_BOMB: -200,
        UNDECIDED: -1000
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def value_function(self, game_state:dict)->float:
    # The value function returns the maximum value of the Q table for a given state, iterating through all the possible actions.
    value = float('-inf')

    for action in ACTIONS:
        if self.q_table[tuple(state_to_features(game_state)), action] > value:
            value = self.q_table[tuple(state_to_features(game_state)), action]

    return value

"""

def optimize(self, n :int) -> list:
    # inizializza i valori a zero
    game_rewards = np.zeros((n,1))

    # fai un round ed estrai il punteggio
    
    punteggio0 = 0

    # inizializza i valori random
    rewards = np.random(n)

    # train

    # estrai il punteggio
    punteggio1 = 0

    # calcola il delta punteggio
    delta_punteggio = 0

    # modifica parametro i di delta_punteggio * 0.1

    # train
    # round
    # delta punteggio
    # parametro 2 ...

    # restituisci i valori finali
"""