# This file is always loaded at the start of a new game.

'''
Called once before the start of a new round.
Use this to initialize everthing you'll need during the game.
The self argument is a persistent object, you can assign values to it which you might need later on.
'''
def setup(self):
    return
    
'''
Called once every step to give your agent the opportunity to pick the best move.
The possible return values are:
• UP
• DOWN
• LEFT
• RIGHT
• BOMB
• WAIT
game_state is a dictionary that defines the current state of the world. 
It contains:
• round (int): Number of rounds since launching of the environment. Starts at 1.
• step (int): Number of steps since beginning of round. Starts at 1.
• field (np.array(width, height)): Describes the tiles of the game world: 1 for crates, -1 for walls and 0 for free tiles.
• bombs ([(int, int), int]): a list of tuples ((x, y), t) for all placed bombs and their countdowns (countdown == 0 means that the bomb is about to explode).
• explosion_map (np.array(width, height)): stating for each tile for how many more steps the explosion will be present (0 where there’s no explosion).
• coins ([(x,y)]): list of coordinates for all currently collectable coins.
• self ((str, int, bool, (int, int))): tuple (n, s, b, (x, y) describing your agent: n is the name, s is the score, b indicates wether it is possible to drop a bomb or not and (x, y) is the position.
• others ([(str, int, bool, (int, int))]): list of tuples similar to the one described above to keep track of the opponents.
• user_input (str/None): user input via GUI.
'''
def act(self, game_state):
    return