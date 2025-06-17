# multiAgents.py
# -----------
# Projeto Minimax - PacMan
# Implementação por Eduardo Gondim Marinho, João Guilherme, Eduardo Serra, Augusto Madi, João Pedro Vieira e Nathan
# Disciplina: Inteligência Artificial
# Curso: Ciência da Computação
# Professor: Nikson Bernardes Fernandes Ferreira
# Data: 17/06/2025

from util import manhattanDistance
from game import Directions
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        Escolhe a melhor ação usando o algoritmo Minimax.
        """
        def minimax(agentIndex=0, depth=0, state=gameState):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = 0 if agentIndex == numAgents - 1 else agentIndex + 1
            nextDepth = depth + 1 if agentIndex == numAgents - 1 else depth

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pac-Man (max)
                bestVal = -float('inf')
                for act in actions:
                    val = minimax(nextAgent, nextDepth,
                                  state.generateSuccessor(agentIndex, act))
                    bestVal = max(bestVal, val)
                return bestVal
            else:  # Fantasmas (min)
                worstVal = float('inf')
                for act in actions:
                    val = minimax(nextAgent, nextDepth,
                                  state.generateSuccessor(agentIndex, act))
                    worstVal = min(worstVal, val)
                return worstVal

        # Ignora STOP para evitar inércia
        legalMoves = [a for a in gameState.getLegalActions(
            0) if a != Directions.STOP]
        if not legalMoves:
            legalMoves = gameState.getLegalActions(0)

        bestScore = -float('inf')
        bestMove = Directions.STOP
        for mv in legalMoves:
            score = minimax(1, 0, gameState.generateSuccessor(0, mv))
            if score > bestScore:
                bestScore = score
                bestMove = mv
        return bestMove


def betterEvaluationFunction(currentGameState: GameState):
    """
    Avaliação baseada em pontuação, comida e distância dos fantasmas.
    Penaliza ficar parado e pequenos loops.
    """
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    distFood = [manhattanDistance(pos, f) for f in foodList]
    minFood = min(distFood) if distFood else 0

    distGhost = [manhattanDistance(pos, g.getPosition()) for g in ghostStates]
    minGhost = min(distGhost) if distGhost else 0

    scaredTimes = [g.scaredTimer for g in ghostStates]
    if min(scaredTimes) > 0:
        minGhost = float('inf')

    score = currentGameState.getScore()
    score -= 1.5 / (minFood + 1)
    score += 2.0 / (minGhost + 1)

    # Penalidade leve se parado
    if currentGameState.getPacmanState().configuration.direction == Directions.STOP:
        score -= 0.5

    # Penalidade leve por movimento em geral (reduz loops)
    score -= 0.05

    return score


# Abreviação para usar no terminal
better = betterEvaluationFunction
