import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from animation import Animation

class Pacman(Entity):
    def __init__(self, nodes, spritesheet):
        Entity.__init__(self, nodes, spritesheet)
        self.name = "pacman"
        self.color = YELLOW
        self.setStartPosition()
        self.lives = 5
        #self.image = self.spritesheet.getImage(0, 1, 32, 32)
        self.startImage = self.spritesheet.getImage(4.5, 0.5, 16, 16)
        self.image = self.startImage
        self.animation = None
        self.animations = {}
        self.defineAnimations()
        self.lifeicons = self.spritesheet.getImage(0, 1, 32, 32)
        
    def reset(self):
        self.setStartPosition()
        self.image = self.startImage
        
    def loseLife(self):
        self.lives -= 1

    def renderLives(self, screen):
        for i in range(self.lives-1):
            x = 10 + 42 * i
            y = TILEHEIGHT * NROWS - 32
            screen.blit(self.lifeicons, (x, y))
            
    def update(self, dt):
        self.visible = True
        self.position += self.direction*self.speed*dt
        self.updateAnimation(dt)
        direction = self.getValidKey()
        if direction:
            self.moveByKey(direction)
        else:
            self.moveBySelf()

    def getValidKey(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return None

    def moveByKey(self, direction):
        if self.direction is STOP:
            if self.node.neighbors[direction] is not None:
                self.target = self.node.neighbors[direction]
                self.direction = direction
        else:
            if direction == self.direction * -1:
                self.reverseDirection()
            if self.overshotTarget():
                self.node = self.target
                self.portal()
                if self.node.neighbors[direction] is not None:
                    if self.node.homeEntrance:
                        if self.node.neighbors[self.direction] is not None:
                            self.target = self.node.neighbors[self.direction]
                        else:
                            self.setPosition()
                            self.direction = STOP
                    else:
                        self.target = self.node.neighbors[direction]
                        if self.direction != direction:
                            self.setPosition()
                            self.direction = direction
                else:
                    if self.node.neighbors[self.direction] is not None:
                        self.target = self.node.neighbors[self.direction]
                    else:
                        self.setPosition()
                        self.direction = STOP

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            d = self.position - pellet.position
            dSquared = d.magnitudeSquared()
            rSquared = (pellet.radius+self.collideRadius)**2
            if dSquared <= rSquared:
                return pellet
        return None

    def eatGhost(self, ghosts):
        for ghost in ghosts:
            d = self.position - ghost.position
            dSquared = d.magnitudeSquared()
            rSquared = (self.collideRadius + ghost.collideRadius)**2
            if dSquared <= rSquared:
                return ghost
        return None

    def eatFruit(self, fruit):
        d = self.position - fruit.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius+fruit.collideRadius)**2
        if dSquared <= rSquared:
            return True
        return False
                                
    def findStartNode(self):
        for node in self.nodes.nodeList:
            if node.pacmanStartNode:
                return node
        return None
    
    def setStartPosition(self):
        self.direction = LEFT
        self.node = self.findStartNode()
        self.target = self.node.neighbors[self.direction]
        self.setPosition()
        self.position.x -= (self.node.position.x - self.target.position.x) / 2


    def defineAnimations(self):
        anim = Animation("static") #keeps the same image wihout reloading all the time
        anim.speed = 1
        anim.addFrame(self.spritesheet.getImage(4.5, 0.5, 16, 16))   #Changed size to fit map
        self.animations["up"] = anim




        #Removed all of this process to make program more processor friendly. Sebastian Grut.

        #anim = Animation("loop")
        #anim.speed = 32
        #anim.addFrame(self.spritesheet.getImage(4, 0, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(1, 0, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(1, 1, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(1, 0, 32, 32))
        #self.animations["right"] = anim
        
        #anim = Animation("loop")
        #anim.speed = 32
        #anim.addFrame(self.spritesheet.getImage(4, 0, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(2, 0, 32, 32))
        #nim.addFrame(self.spritesheet.getImage(2, 1, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(2, 0, 32, 32))
        #self.animations["down"] = anim
        
        #anim = Animation("loop")
        #anim.speed = 32
        #anim.addFrame(self.spritesheet.getImage(4, 0, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(3, 0, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(3, 1, 32, 32))
        #anim.addFrame(self.spritesheet.getImage(3, 0, 32, 32))
        #self.animations["up"] = anim


        #Fine to leave for tiem being. Death animation removed in run anyway.

        anim = Animation("once")
        anim.speed = 10
        anim.addFrame(self.spritesheet.getImage(0, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(1, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(2, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(3, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(4, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(5, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(6, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(7, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(8, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(9, 7, 32, 32))
        anim.addFrame(self.spritesheet.getImage(10, 7, 32, 32))
        self.animations["death"] = anim
        
        anim = Animation("static")
        anim.addFrame(self.spritesheet.getImage(4, 0, 32, 32))
        self.animations["idle"] = anim
        
    def updateAnimation(self, dt):
        if True:    #from self.direction == UP
            self.animation = self.animations["up"] #More processor friendly to only have one function.

        #elif self.direction == DOWN:
        #    self.animation = self.animations["down"]
        #elif self.direction == LEFT:
        #    self.animation = self.animations["left"]
        #elif self.direction == RIGHT:
        #    self.animation = self.animations["right"]
        #elif self.direction == STOP:
        #    self.animation = self.animations["idle"]
        self.image = self.animation.update(dt)
            
