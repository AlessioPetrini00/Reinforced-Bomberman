# Only loaded when the environment is launched in training mode.

# Called right after the setup call in callbacks.py.
# Use this to initialize the variables you only need for training.
def setup_training(self):
    return

'''
Called once after each step except the last.
Use this to collect training data into the experience buffer.
events is a list of things that happend in the game, the possibilities are:
• MOVED_LEFT
• MOVED_RIGHT
• MOVED_UP
• MOVED_DOWN
• WAITED: intentionally didn’t act.
• INVALID_ACTION: picked a non-existing acttion or one that couldn’t be executed.
• BOMB_DROPPED
• BOMB_EXPLODED
• CRATE_DESTROYED: by own bomb.
• COIN_FOUND: by own bomb.
• COIN_COLLECTED
• KILLED_OPPONENT
• KILLED_SELF
• GOT_KILLED
• OPPONENT_ELIMINATED
• SURVIVED_ROUND
'''
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    return

# Similar to the previous one but called only after the last step.
def end_of_round(self, last_game_state, last_action, events):
    return