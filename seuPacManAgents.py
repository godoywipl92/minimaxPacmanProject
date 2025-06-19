# multiAgents.py
# -----------
# Projeto Minimax - PacMan
# Implementação por Isaac Machado, Gustavo Godoy, Gabriel Henrique, Andrei Barone, Gabriel Henrique
# Disciplina: Inteligência Artificial
# Curso: Ciência da Computação
# Professor: Nikson Bernardes Fernandes Ferreira
# Data: 17/06/2025

from util import manhattanDistance
from game import Directions
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='betterEvaluationFunction', depth='3'):
        super().__init__(evalFn, depth)
        self.recentPositions = []  # Rastreia posições recentes para evitar loops
        self.maxRecentPositions = 10  # Máximo de posições para rastrear
    
    def getAction(self, gameState: GameState):
        """
        Escolhe a melhor ação usando o algoritmo Minimax.
        Esta função implementa o algoritmo Minimax recursivo para tomar decisões
        estratégicas no jogo Pac-Man, considerando os movimentos dos fantasmas.
        """
        # Atualiza posições recentes
        currentPos = gameState.getPacmanPosition()
        self.recentPositions.append(currentPos)
        if len(self.recentPositions) > self.maxRecentPositions:
            self.recentPositions.pop(0)
        
        def minimax(agentIndex=0, depth=0, state=gameState):
            """
            Função recursiva que implementa o algoritmo Minimax.
            
            Args:
                agentIndex: Índice do agente atual (0 = Pac-Man, 1+ = fantasmas)
                depth: Profundidade atual na árvore de busca
                state: Estado atual do jogo
            
            Returns:
                float: Valor da utilidade do estado (para nós internos)
                ou ação (para o nó raiz)
            """
            # verifica se jogo acabou, se sim retorna self.evaluationFunction(state) ou betterEvaluationFunction do estado
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # calcula próximo agente
            numAgents = state.getNumAgents()
            nextAgent = 0 if agentIndex == numAgents - 1 else agentIndex + 1
            
            # calcula próxima profundidade (apenas se agentIndex == self.index a profundidade aumenta)
            nextDepth = depth + 1 if agentIndex == numAgents - 1 else depth

            # para cada ação possível em state.getLegalActions(agentIndex)
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pac-Man (max)
                # se for um passo de maximização e o score for maior que o anterior, selecione ele
                bestVal = -float('inf')
                for action in actions:
                    # calcula o próximo estado com state.generateSuccessor(agentIndex, action)
                    nextState = state.generateSuccessor(agentIndex, action)
                    # calcula o score chamando minimax recursivamente
                    score = minimax(nextAgent, nextDepth, nextState)
                    if score > bestVal:
                        bestVal = score
                return bestVal
            else:  # Fantasmas (min)
                # se for um passo de minimização e o score for menor que o anterior, selecione ele
                worstVal = float('inf')
                for action in actions:
                    # calcula o próximo estado com state.generateSuccessor(agentIndex, action)
                    nextState = state.generateSuccessor(agentIndex, action)
                    # calcula o score chamando minimax recursivamente
                    score = minimax(nextAgent, nextDepth, nextState)
                    if score < worstVal:
                        worstVal = score
                return worstVal

        # Ignora STOP para evitar inércia
        legalMoves = [a for a in gameState.getLegalActions(0) if a != Directions.STOP]
        if not legalMoves:
            legalMoves = gameState.getLegalActions(0)

        bestScore = -float('inf')
        bestMove = Directions.STOP
        
        # Avalia todas as ações possíveis do Pac-Man
        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            score = minimax(1, 0, nextState)
            
            # Penalização por movimentos que levam a posições recentes (evita loops)
            nextPos = nextState.getPacmanPosition()
            if nextPos in self.recentPositions[-3:]:  # Se a próxima posição foi visitada recentemente
                score -= 100  # Penalização para evitar loops
            
            if score > bestScore:
                bestScore = score
                bestMove = move
                
        return bestMove


def betterEvaluationFunction(currentGameState: GameState):
    """
    Função de avaliação otimizada para priorizar a busca ativa por comida.
    O Pac-Man deve sempre tentar comer o máximo possível, assumindo riscos calculados.
    """
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    layout = currentGameState.data.layout
    score = currentGameState.getScore()

    # PRIORIDADE MÁXIMA: Comida
    numFood = len(foodList)
    
    # Penalização forte por comida restante (incentiva comer tudo)
    score -= numFood * 50  # Reduzido de 100 para 50 para evitar loops
    
    # Bônus alto quando restam poucas comidas
    if numFood <= 5:
        score += 300  # Reduzido de 500 para 300
    if numFood <= 3:
        score += 500  # Reduzido de 800 para 500
    if numFood == 1:
        score += 800  # Reduzido de 1200 para 800
    
    # Incentivo para buscar comida mais próxima (evita loops)
    if foodList:
        foodDistances = [manhattanDistance(pos, food) for food in foodList]
        minFoodDist = min(foodDistances)
        
        # Penalização por distância da comida (mais suave)
        score -= minFoodDist * 2  # Reduzido de 3 para 2
        
        # Bônus por estar próximo da comida
        if minFoodDist <= 3:
            score += 100  # Reduzido de 200 para 100
        if minFoodDist <= 1:
            score += 200  # Reduzido de 400 para 200
        
        # Incentivo para explorar diferentes áreas
        # Calcula a média das distâncias para todas as comidas
        avgFoodDist = sum(foodDistances) / len(foodDistances)
        if avgFoodDist > minFoodDist + 5:  # Se há comida muito mais próxima que a média
            score += 150  # Bônus extra para ir para a comida mais próxima
    else:
        score += 1000  # Reduzido de 2000 para 1000

    # SEGUNDA PRIORIDADE: Fantasmas (evita loops perigosos)
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pos, ghostPos)
        
        if ghostState.scaredTimer > 0:
            # Fantasmas assustados - incentiva a perseguição
            if ghostDist <= 1:
                score += 300  # Reduzido de 500 para 300
            elif ghostDist <= 3:
                score += 150 / ghostDist  # Reduzido de 300 para 150
        else:
            # Fantasmas normais - penalização mais forte para evitar loops perigosos
            if ghostDist <= 1:
                score -= 1000  # Aumentado de 800 para 1000
            elif ghostDist <= 2:
                score -= 200 / ghostDist  # Aumentado de 100 para 200
            elif ghostDist <= 4:
                score -= 50 / ghostDist  # Aumentado de 20 para 50

    # TERCEIRA PRIORIDADE: Cápsulas (incentivo moderado)
    if capsules:
        minCapsuleDist = min([manhattanDistance(pos, capsule) for capsule in capsules])
        if minCapsuleDist <= 5:
            score -= minCapsuleDist * 3  # Reduzido de 5 para 3
        else:
            score -= minCapsuleDist * 1.5  # Reduzido de 2 para 1.5

    # Penalidade por ficar parado
    if currentGameState.getPacmanState().configuration.direction == Directions.STOP:
        score -= 50  # Reduzido de 100 para 50

    # Penalização por tempo (moderada)
    score -= 2  # Reduzido de 5 para 2

    # Penalização por estar em áreas muito exploradas
    # Calcula se está muito próximo das bordas (pode indicar loop)
    edgePenalty = 0
    if pos[0] <= 2 or pos[0] >= layout.width - 3:
        edgePenalty += 20
    if pos[1] <= 2 or pos[1] >= layout.height - 3:
        edgePenalty += 20
    
    # Bônus por estar no centro (incentiva exploração)
    centerX, centerY = layout.width // 2, layout.height // 2
    distanceFromCenter = manhattanDistance(pos, (centerX, centerY))
    if distanceFromCenter < 5:
        score += 20  # Aumentado de 10 para 20
    
    # Penalização por estar muito tempo nas bordas
    score -= edgePenalty

    # Incentivo para movimento direcional (evita oscilação)
    # Se há comida em uma direção específica, incentiva ir nessa direção
    if foodList:
        # Encontra a direção da comida mais próxima
        closestFood = min(foodList, key=lambda f: manhattanDistance(pos, f))
        dx = closestFood[0] - pos[0]
        dy = closestFood[1] - pos[1]
        
        # Pequeno bônus se está se movendo na direção da comida
        currentDir = currentGameState.getPacmanState().configuration.direction
        if (dx > 0 and currentDir == Directions.EAST) or \
           (dx < 0 and currentDir == Directions.WEST) or \
           (dy > 0 and currentDir == Directions.NORTH) or \
           (dy < 0 and currentDir == Directions.SOUTH):
            score += 30

    return score


# Abreviação para usar no terminal
better = betterEvaluationFunction
