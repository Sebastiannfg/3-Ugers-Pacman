#Pacman integrated into AI program by Sebastian Grut


import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pauser
from levels import LevelController
from text import TextGroup
from sprites import Spritesheet
from maze import Maze
import atari_py
import PIL
import matplotlib

#AI program imports
import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import time
import random
from gym import error, spaces
from gym import utils
from gym.utils import seeding
#AI constants
# An episode is a full game

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

train_episodes = 300
test_episodes = 100


#PACMAN PROGRAM CODE --------------------------------------------------------------------------------------
#PACMAN PROGRAM CODE --------------------------------------------------------------------------------------




class GameController(object):

    #__init__ function
    def __init__(self):
        #Added for AI program from atari_env.py
        self.ale = atari_py.ALEInterface()
        self.last_score = 0


        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.setBackground()
        self.clock = pygame.time.Clock()
        self.pelletsEaten = 0
        self.fruit = None
        self.pause = Pauser(True)
        self.level = LevelController()
        self.text = TextGroup()
        self.sheet = Spritesheet()
        self.maze = Maze(self.sheet)

        #Taken from atari program
        #self.action_space = spaces.Discrete(len(self._action_set))
        self.seed()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREENHEIGHT, SCREENWIDTH, 3), dtype=np.uint8)



    #All code below is made using atari_env.py from the gym pack. in the gym pack
    def move(self, action):
        if action == 0:
            self.direction = UP
        elif action == 1:
            self.direction = DOWN
        elif action == 2:
            self.direction = LEFT
        elif action == 3:
            self.direction = RIGHT
        self.Update()

    def step(self, action):
        self.move(action)
        reward = self.score-self.last_score

        #rewards for winning or losing
        if self.gameover:
            reward -= 1000
        elif self.is_done:
            reward += 1000
        done = self.is_done()
        self.last_score = self.score

        observation = self._get_obs()
        return observation, reward, done, {}


    def is_done(self):
        if (self.pellets.isEmpty() or self.gameover):
            return True
        else:
            return False

    def _get_image(self):
        return pygame.surfarray.array3d(self.screen)

    def _get_obs(self):
        img = self._get_image()
        return img

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)

        return [seed1, seed2]




    #general activation




    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def reset(self):   #was StartGame, changed to fit with AI program
        #Sets up the game

        self.level.reset()
        levelmap = self.level.getLevel()
        self.maze.getMaze(levelmap["mazename"].split(".")[0])
        self.maze.constructMaze(self.background)
        self.nodes = NodeGroup(levelmap["mazename"])
        self.pellets = PelletGroup(levelmap["pelletname"])
        self.pacman = Pacman(self.nodes, self.sheet)
        #self.ghosts = GhostGroup(self.nodes, self.sheet)  Ghost command, removed to get rid of ghosts
        self.pelletsEaten = 0
        self.fruit = None
        self.pause.force(False)#Was True, set to false for automatic start
        self.gameover = False
        self.score = 0
        #self.text.showReady()    #Removed, so that we don't have Ready permanently on screen
        #self.text.updateLevel(self.level.level+1)  #Removed to start same level each time
        
    def startLevel(self):
        #Begins the given level

        levelmap = self.level.getLevel()
        self.setBackground()
        self.nodes = NodeGroup(levelmap["mazename"])
        self.pellets = PelletGroup(levelmap["pelletname"])
        self.pacman.nodes = self.nodes
        self.pacman.reset()
        #self.ghosts = GhostGroup(self.nodes, self.sheet)   Ghost command, removed to get rid of ghosts
        self.pelletsEaten = 0
        self.fruit = None
        self.pause.force(True)
        self.text.showReady()
        self.text.updateLevel(self.level.level+1)

    def restartLevel(self):
        #Restarts the game.
        self.pacman.reset()
        #self.ghosts = GhostGroup(self.nodes, self.sheet) Ghost command, removed to get rid of ghosts
        self.fruit = None
        self.pause.force(True)
        self.text.showReady()
    p = 0
    refresh = 60
    speed = 1000
    def Update(self):
        #Where the game instances are run #################################################################
        if not self.gameover:
            dt = self.clock.tick(GameController.refresh) / GameController.speed
            # CHANGE TIME  (   clock.tick( refresh rate )   / speed (lower number, higher speed)
            #NOTE: Refresh rate controls how often p is increased. Must be used carefully if - per turn is consistent.
            if not self.pause.paused:

                self.pacman.update(dt, self.direction)
                #self.ghosts.update(dt, self.pacman)   Stopper spøgelser med at opdaterer
                #GameController.p +=1
                #print(GameController.p)
                #if GameController.p == 500:
                #    self.score -= 1
                #    GameController.p = 0
                # This ^^^^^ was added to reduce the score over time ######################################################

                if self.fruit is not None:
                    self.fruit.update(dt)
                if self.pause.pauseType != None:
                    self.pause.settlePause(self)
                self.checkPelletEvents()
                #self.checkGhostEvents()   Ghost command, removed to get rid of ghosts
                self.checkFruitEvents()

            self.pause.update(dt)
            self.pellets.update(dt)
            self.text.update(dt)
        #self.check_Events()
        self.text.updateScore(self.score)
        #env.render     the AI program does this


    def check_Events(self): #Was checkEvents. Changed to match AI program
        #Simple commands while in game

        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif self.gameover:
                self.startGame()
            #Spacebar is removed, and instead game just restarts automatically


            #elif event.type == KEYDOWN:
            #    if event.key == K_SPACE:
            #        if self.gameover:
            #            self.startGame()
            #        else:
            #            self.pause.player()
            #            if self.pause.paused:
            #               self.text.showPause()
            #            else:
            #                self.text.hideMessages()

    def checkPelletEvents(self):
        #This section is in charge of checking if Pacman is on a pellet.


        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pelletsEaten += 1
            self.score += pellet.points
            if (self.pelletsEaten == 70 or self.pelletsEaten == 140):
                if self.fruit is None:
                    self.fruit = Fruit(self.nodes, self.sheet)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == "powerpellet":
                pass
                #self.ghosts.resetPoints()   Ghost command, removed to get rid of ghosts
                #self.ghosts.freightMode()   Ghost command, removed to get rid of ghosts
            if self.pellets.isEmpty():
                self.pacman.visible = False
                #self.ghosts.hide()    Ghost command, removed to get rid of ghosts
                self.pause.startTimer(0.1, "clear") #Used to be 3, has been set to 0.1. No reset timer.
                
    def checkGhostEvents(self):
        #In charge of ghosts

        self.ghosts.release(self.pelletsEaten)
        ghost = self.pacman.eatGhost(self.ghosts)
        if ghost is not None:
            if ghost.mode.name == "FREIGHT":
                self.score += ghost.points
                self.text.createTemp(ghost.points, ghost.position)
                self.ghosts.updatePoints()
                ghost.spawnMode(speed=2)
                self.pause.startTimer(1)
                self.pacman.visible = False
                ghost.visible = False
            elif ghost.mode.name == "CHASE" or ghost.mode.name == "SCATTER":
                self.pacman.loseLife()
                self.ghosts.hide()
                self.pause.startTimer(0.1, "die")   #Death timer, set to 0 to optimize time.

    def checkFruitEvents(self):
        #basically in charge of fruits, in same way as above.

        if self.fruit is not None:
            if self.pacman.eatFruit(self.fruit):
                self.score += self.fruit.points
                self.text.createTemp(self.fruit.points, self.fruit.position)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None
            #if self.pacman.eatFruit(self.fruit) or self.fruit.destroy:
                #self.fruit = None

    def resolveDeath(self):
        #Checks if the game needs to end by keeping track of death tol

        if self.pacman.lives == 0:
            self.gameover = False  #This should be true, but has been changed to remove lives from the game
        else:
            self.restartLevel()
        self.pause.pauseType = None

    def resolveLevelClear(self):

        #self.level.nextLevel()
        #self.startLevel()

        self.startGame()
        self.pause.pauseType = None



    def render(self):
        #In charge of rendering all characters, ghosts, pellets, ghosts etc...

        self.screen.blit(self.background, (0, 0))
        #self.nodes.render(self.screen)  (This change was made by the creators doing)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        #self.ghosts.render(self.screen)    Ghost command, removed to get rid of ghosts
        self.pacman.renderLives(self.screen)
        self.text.render(self.screen)


        pygame.display.update()



#AI PROGRAM CODE --------------------------------------------------------------------------------------
#AI PROGRAM CODE --------------------------------------------------------------------------------------




def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return


    batch_size = 32
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([encode_observation(transition[0], env.observation_space.shape) for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([encode_observation(transition[3], env.observation_space.shape) for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(encode_observation(observation, env.observation_space.shape))
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

def encode_observation(observation, n_dims):
    return observation
def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the timepygame.surfarray.array3d(self.screen)speed
    decay = 0.01
    frame = 0
    # 1. Initialize the Target and Main models
    # Main Model (updated every step)
    model = agent(env.observation_space.shape, env.action_space.n)
    # Target Model (updated every 100 steps)
    target_model = agent(env.observation_space.shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):

        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1

            if True:
                env.render()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = encode_observation(observation, env.observation_space.shape[0])
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    env.close()


#AI import code for pacman game. To be replaced
RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

#env = gym.make('MsPacman-ram-v0')
env = GameController()     #makes the GameController our class
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(env.seed(RANDOM_SEED))

if __name__ == '__main__':
    main()

#if __name__ == "__main__":
#    #starts and runs the game
#   game = GameController()
#   game.reset()
#   while True:
#       game.Update()

