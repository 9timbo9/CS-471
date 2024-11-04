# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #my agent seems to stop when there 1 food left so im making it a strong punishment to stop
        # if action == 'Stop':
        #     return -float('inf')

        newFood = newFood.asList()

        LARGE_POSITIVE = float('inf')

        if len(newFood) == 0:
            return LARGE_POSITIVE

        dis_from_ghost = []
        for ghosts in newGhostStates:
            dis_from_ghost.append(manhattanDistance(newPos,ghosts.getPosition()))
        
        dis_from_food = []
        for food in newFood:
            dis_from_food.append(manhattanDistance(newPos,food))
        

        
        ghostPenalty = 0
        if min(dis_from_ghost) > 0:
            ghostPenalty = 1.0 /max(min(dis_from_ghost), 1)

        foodReward = 10.0 / max(min(dis_from_food), 1)  

        
        #This orientation for the return passed all the test for me
        return successorGameState.getScore() + foodReward - ghostPenalty * 4

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def terminal(state,depth):
              return state.isWin() or state.isLose() or depth == self.depth
        #terminal test

        def minimax(state,depth,index): #minimax representation from the psuedocode from slides
            if terminal(state,depth):
                return self.evaluationFunction(state)
            if index == 0:
                return max_value(state,depth)
            else:
                return min_value(state,depth,index)           
 
        def max_value(state,depth):
            v = -float('inf')
            for action in state.getLegalActions(0): #only 1 max node, so we don't need index as a parameter
                v = max(v, minimax(state.generateSuccessor(0,action),depth,1)) 
            return v

        def min_value(state,depth,index):
            v = float('inf')
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)

                if index == state.getNumAgents() - 1: #if its the last ghost go to next level/ max node
                    v = min(v, minimax(successor,depth+1,0))
                else:
                    v = min(v, minimax(successor, depth, index + 1)) #else go to the next ghost
            return v

            
        move = None
        v = -float('inf')
        for action in gameState.getLegalActions(0):  # inital index is 0
            successor = gameState.generateSuccessor(0, action)
            tempv = minimax(successor, 0, 1)  
            if tempv > v:
                v = tempv 
                move = action

        return move




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def alphabeta(state,depth,index,alpha,beta): #minimax representation from the psuedocode from slides
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if index == 0:
                return max_value(state,depth,index,alpha,beta)
            else:
                return min_value(state,depth,index,alpha,beta)           
 
        def max_value(state,depth,index,alpha,beta):
            v = -float('inf')
            for action in state.getLegalActions(index): #only 1 max node, so we don't need index as a parameter
                successor = state.generateSuccessor(index, action)
                v = max(v, alphabeta(successor,depth,index+1,alpha,beta))
                if v > beta:
                    return v
                alpha = max(alpha,v)
            return v

        def min_value(state,depth,index,alpha,beta):
            v = float('inf')
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                if index == state.getNumAgents() - 1: #if its the last ghost go to next level/ max node
                    v = min(v, alphabeta(successor,depth+1,0,alpha,beta))
                else:
                    v = min(v, alphabeta(successor, depth, index + 1,alpha,beta)) #else go to the next ghost
                if v < alpha: # exiting the loop to prune
                    return v
                beta = min(beta, v)
            return v

            
        move = None
        v = -float('inf')
        alpha, beta = -float('inf'), float('inf')
        for action in gameState.getLegalActions(0):  # initial index is 0
            successor = gameState.generateSuccessor(0, action)
            tempv = alphabeta(successor, 0, 1, alpha, beta)
            if tempv > v:
                v = tempv
                move = action
            alpha = max(alpha, v)


        return move

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def terminal(state,depth):
              return state.isWin() or state.isLose() or depth == self.depth
        #terminal test

        def expectimax(state,depth,index): #expectimax came mostly from minimax
            if terminal(state,depth):
                return self.evaluationFunction(state)
            if index == 0:
                return max_value(state,depth)
            else:  # ghost's turn 
                return exp_value(state, depth, index)
         
 
        def max_value(state,depth):
            v = -float('inf')
            for action in state.getLegalActions(0): #only 1 max node, so we don't need index as a parameter
                v = max(v, expectimax(state.generateSuccessor(0,action),depth,1)) 
            return v    

        def exp_value(state,depth,index): #from psuedo code in slide
            actions = state.getLegalActions(index)
            probability = 1 / len(actions)
            expectedValue = 0
            for action in actions:
                successor = state.generateSuccessor(index, action)
                if index == state.getNumAgents() - 1: #if its the last ghost go to next level/ max node
                    value = expectimax(successor, depth + 1, 0)  # Back to Pacman
                else:
                    value = expectimax(successor, depth, index + 1)  # Next ghost
                expectedValue += probability * value
            return expectedValue

            
        move = None
        v = -float('inf')
        for action in gameState.getLegalActions(0):  # inital index is 0
            successor = gameState.generateSuccessor(0, action)
            tempv = expectimax(successor, 0, 1)  
            if tempv > v:
                v = tempv 
                move = action

        return move
    
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <Almost the same as the initial evaluation function but now I take advantage of the ghosts scareTimer>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore() 

    dis_to_food = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(dis_to_food) > 0: # added this to prevent an empty min() arguement error 
        score += 10.0 / min(dis_to_food)

    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0: #prioritize when ghosts are scare
                score += 100 / distance
            else:  
                score += -10.0 / distance
        else:
            return -100000000.0 #the game ends at this point

    return score

better = betterEvaluationFunction
