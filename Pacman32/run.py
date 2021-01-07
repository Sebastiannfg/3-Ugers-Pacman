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

class GameController(object):
    #general activation
    def __init__(self):
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
        
    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def startGame(self):
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
        
    def update(self):
        #Where the game instances are run

        if not self.gameover:
            dt = self.clock.tick(300) / 300.0
            # CHANGE TIME  (   clock.tick( refresh rate )   / speed (lower number, higher speed)
            if not self.pause.paused:
                self.pacman.update(dt)
                #self.ghosts.update(dt, self.pacman)   Stopper spøgelser med at opdaterer
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
        self.checkEvents()
        self.text.updateScore(self.score)
        self.render()

    def checkEvents(self):
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
        #self.nodes.render(self.screen)  (Not my doing)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        #self.ghosts.render(self.screen)    Ghost command, removed to get rid of ghosts
        self.pacman.renderLives(self.screen)
        self.text.render(self.screen)
        pygame.display.update()

#
if __name__ == "__main__":
    #starts and runs the game

    game = GameController()
    game.startGame()
    while True:
        game.update()
