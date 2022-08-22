'''
Date: 10/30/2021
Course: Foundations of Artificial Intelligence (CS 5100)
Project: Programming Project 2: Multi - Agent Search
Group 5: Artik Bharoliya, Kaitlyn Lowen, Marjan Gohari, Monideep Chakraborti, Zhengtian Zhang
'''
from util import manhattanDistance
from game import Directions
import random, util
import math

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
        # print(newFood.asList())

        # PART 1: collect information on items remaining in the game
        # 1. amount of food in successor state
        food_available = newFood.asList()
        food_in_successor = len(food_available)

        # 2. amount of food at current state
        food_current_state = currentGameState.getFood().asList()
        current_amount_of_food = len(food_current_state)

        # 3. power ups in game
        power_up_tokens = successorGameState.getCapsules()
        power_ups_available = len(power_up_tokens)

        # 4. set score to 0
        my_score = 0

        # 5. Scared times
        scared_time = 0
        for token in newScaredTimes:
            scared_time += token


        # PART 2: Distance calculations

        # using the manhattan distance provided, find the MD to each ghost from SUCCESSOR position
        successor_distance_to_ghost = []
        for single_ghost in newGhostStates:
            # position_of_ghost.append(single_ghost.getPosition())
            # calculate ghost distance from successor game state position
            ghost_distance = manhattanDistance(newPos, single_ghost.getPosition())
            successor_distance_to_ghost.append(ghost_distance)

        # using the manhattan distance provided, find the MD to each ghost from CURRENT position
        current_distance_to_ghost = []
        for ghost in currentGameState.getGhostStates():
            # position_of_ghost.append(single_ghost.getPosition())
            # calculate ghost distance from successor game state position
            current_ghost_distance = manhattanDistance(newPos, ghost.getPosition())
            current_distance_to_ghost.append(current_ghost_distance)

        # using the MD, calculate the distances to each food item in successor position
        distance_to_food = []
        for each_food in food_available:
            # calculate distance from successor game state to food item
            food_distance = manhattanDistance(each_food, newPos)
            distance_to_food.append(food_distance)

        # PART 3: determine score
        # set distance to infinity
        smallest_distance_food = math.inf
        if (food_in_successor == 0):
            smallest_distance_food = 0

        for i in range(len(distance_to_food)):
            current_dist_food = distance_to_food[i] + food_in_successor * 100
            if (current_dist_food < smallest_distance_food):
                smallest_distance_food = current_dist_food
        my_score = -smallest_distance_food

        for ghosts in current_distance_to_ghost:
            if (ghosts <= 1):
                # if ghosts are close, penalize the score until pacman chooses to not go here
                my_score -= 1000

        return my_score
        # return successorGameState.getScore()

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

        '''
        Minimax algorithm:
        gameState: current state of the Pacman game
        depth: how many moves ahead do we want to search
        agent: Pacman is always agent 0, the ghosts will be >= 1
        '''
        
        def mini_max(gameState, depth, agent):

            # Terminal state (base case): if the depth defined is reached or the game is finished (Pacman won or lost)
            if depth == self.depth or gameState.isLose() or gameState.isWin(): 
                return self.evaluationFunction(gameState)

            # Process for maximizer's (Pacman) turn    
            if agent == 0:
                # Maximize the returns for Pacman and pass the turn to agent 1 (Ghost 1), here the value of agent == 0
                return max(mini_max(gameState.generateSuccessor(agent, actions), depth, 1) for actions in gameState.getLegalActions(agent))

            # Process for multiple minimizer's (Ghosts) turn    
            else:  
                # index of the next agent
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                
                # If the agent = 0, this means that all ghosts have played their turns, increase depth and continue playing
                if nextAgent == 0:
                   depth += 1
                # Minimize the returns for the ghosts and pass the turn to the next ghost or Pacman   
                return min(mini_max(gameState.generateSuccessor(agent, actions), depth, nextAgent) for actions in gameState.getLegalActions(agent))

        '''
        Previously, we created a helper miniMax function for the moves for maximizer and the minimizer. 
        Now, we focus on the root node: Pacman. The Pacman is agent 0 and will always be the maximizer.
        We shall use the previously created helper function to find the moves for Pacman from the start. 
        '''
        # initialize to - infinity
        maxEval = float("-inf")

        # for Pacman, agent == 0
        for agentState in gameState.getLegalActions(0):

            # here depth = self.index and the next turn will be given to agent 1 (Ghost 1)
            evalValue = mini_max(gameState.generateSuccessor(0, agentState), 0, 1)
            if evalValue > maxEval or maxEval == float("-inf"):
                maxEval = evalValue
                action = agentState

        # return the minimax action from the current game state                
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        current_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        pacman_action = Directions.STOP

        possible_actions = gameState.getLegalActions(0).copy()

        for next_action in possible_actions:
            nextState = gameState.generateSuccessor(0, next_action)

            next_value = self.get_node_value(nextState, 0, 1, alpha, beta)
            if next_value > current_value:
                current_value, pacman_action = next_value, next_action
            alpha = max(alpha, current_value)
            
        return pacman_action


    def get_node_value(self, gameState, cur_depth=0, agent_index=0, alpha=-math.inf, beta=math.inf):
        """
        this method decides what should be calculated - alpha, beta or evaluation function.
        """
        maximums = [0] 
        minimums = list(range(1, gameState.getNumAgents()))

        if cur_depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        elif agent_index in maximums:
            return self.alpha_value(gameState, cur_depth, agent_index, alpha, beta)
        elif agent_index in minimums:
            return self.beta_value(gameState, cur_depth, agent_index, alpha, beta)
        else:
            print('Errors occur in your party division !!! ')
    
    def alpha_value(self, gameState, cur_depth, agent_index, alpha=-math.inf, beta=math.inf):
        '''
        Calculates the alpha value i.e. Max's best move on path to root.
        '''
        v = -math.inf
        possible_actions = gameState.getLegalActions(agent_index)
        for index, action in enumerate(possible_actions):
            next_v = self.get_node_value(gameState.generateSuccessor(agent_index, action),
                                         cur_depth, agent_index + 1, alpha, beta)
            v = max(v, next_v)
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def beta_value(self, gameState, cur_depth, agent_index, alpha=-math.inf, beta=math.inf):
        
        '''
        Calculates the beta value i.e. Min's best move on path to root.
        '''
        
        v = math.inf
        possible_actions = gameState.getLegalActions(agent_index)
        for index, action in enumerate(possible_actions):
            if agent_index == gameState.getNumAgents() - 1:
                next_v = self.get_node_value(gameState.generateSuccessor(agent_index, action),
                                             cur_depth + 1, 0, alpha, beta)
                v = min(v, next_v)  
                if v < alpha:
                    return v
            else:
                next_v = self.get_node_value(gameState.generateSuccessor(agent_index, action),
                                             cur_depth, agent_index + 1, alpha, beta)
                v = min(v, next_v)  
                if v < alpha:  
                    
                    return v
            beta = min(beta, v)
        return v

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
        return self.maximizer(gameState, 0, 0)
        util.raiseNotDefined()

    def maximizer(self, gameState, depth, agent):
        
        # Terminate if the depth defined is reached or Pacman won or lost
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Initialize the max value
        max = float("-inf")

        # Get all legal actions for the Pacman
        legalActions = gameState.getLegalActions(agent)
        
        for action in legalActions:
            # Implement Expectimax for every Ghost
            value = self.expecter(gameState.generateSuccessor(agent, action), depth, 1)
            if value > max:
                max = value
                move = action
        if depth is 0 and agent is 0:
            return move
        else:
            return max

    def expecter(self, gameState, depth, agent):
        
        # Terminate if the depth defined is reached or Pacman won or lost
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
            
        # Initialize the value
        value = 0

        # Get all legal actions for the Ghost
        legalActions = gameState.getLegalActions(agent)
        
        for action in legalActions:
            # If this is the last Ghost, calculate the Max for the Pacman
            if agent == gameState.getNumAgents() - 1:
                value += self.maximizer(gameState.generateSuccessor(agent, action), depth + 1, 0)
            else:
                # Add the value for the next Ghost
                value += self.expecter(gameState.generateSuccessor(agent, action), depth, agent + 1)
        # Calculate average value
        return value / float(len(legalActions))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evluation function (question 5). DESCRIPTION: <write something here so we know what you did>
    """
    
    """* YOUR CODE HERE *"""
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf') 
    pacman_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_time = [ghostState.scaredTimer for ghostState in ghost_states] 
    foodcount = len(food_pos.asList())
    if foodcount == 0:
        return float('inf')
    minfoodvalue = float('inf')
    for food in food_pos.asList():
        minfoodvalue = min(minfoodvalue, manhattanDistance(food, pacman_pos)) 
    minghostvalue = float('inf')
    for ghost in ghost_states:
        minghostvalue = min(minghostvalue, manhattanDistance(ghost.getPosition(), pacman_pos))
    minScaredTimes = min(scared_time) 
    if minScaredTimes == 0:
        if minghostvalue < 3:
            return float('-inf')
        else:
            return currentGameState.getScore() + (1.0 / minfoodvalue) + (1.0 / foodcount)
    else:
        return currentGameState.getScore() + (1.0 / minfoodvalue) + minScaredTimes + (1.0 / minghostvalue) + (1.0 / foodcount)

# Abbreviation
better = betterEvaluationFunction