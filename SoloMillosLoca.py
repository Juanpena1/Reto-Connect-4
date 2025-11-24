import numpy as np
from connect4.policy import Policy


class Hello(Policy):

    def __init__(self):
        self.mcts_iterations = 25  # Iteraciones iniciales
        self.exploration_weight = 1.5
    
    def mount(self) -> None:
        pass
    
    def act(self, s: np.ndarray) -> int:
        available_cols = [c for c in range(7) if s[0, c] == 0]
        
        if len(available_cols) == 0:
            return -1
        if len(available_cols) == 1:
            return available_cols[0]
        
        # Para decisiones simples, usar heurística
        immediate_action = self._check_immediate_actions(s, available_cols)
        if immediate_action is not None:
            return immediate_action
        
        # Para decisiones complejas, usar MCTS
        return self._basic_mcts(s, available_cols)
    
    def _check_immediate_actions(self, state, available_cols):
        """Heurísticas de Alex para decisiones rápidas"""
        for col in available_cols:
            if self._would_win(state, col, 1):
                return col
        for col in available_cols:
            if self._would_win(state, col, -1):
                return col
        return None
    
    def _basic_mcts(self, state, available_cols):
        """Implementación inicial de MCTS"""
        visits = [0] * 7
        wins = [0] * 7
        
        for i in range(self.mcts_iterations):
            # Selección con UCB1
            action = self._ucb_selection(available_cols, visits, wins)
            
            # Simulación
            result = self._simulate(state, action)
            
            # Actualización
            visits[action] += 1
            if result > 0:
                wins[action] += 1
        
        # Elegir mejor acción basada en visitas
        best_action = max(available_cols, key=lambda col: visits[col])
        return best_action
    
    def _ucb_selection(self, available_cols, visits, wins):
        """Algoritmo UCB1 para balance exploración/explotación"""
        total_visits = sum(visits)
        
        # Priorizar acciones no exploradas
        for col in available_cols:
            if visits[col] == 0:
                return col
        
        # Calcular UCB1 para acciones exploradas
        best_score = -1
        best_action = available_cols[0]
        
        for col in available_cols:
            win_rate = wins[col] / visits[col]
            explore = self.exploration_weight * np.sqrt(np.log(total_visits) / visits[col])
            score = win_rate + explore
            
            if score > best_score:
                best_score = score
                best_action = col
        
        return best_action
    
    def _simulate(self, state, first_action):
        """Simulación básica del juego"""
        current_state = state.copy()
        current_player = 1
        
        # Aplicar primera acción
        current_state = self._drop_piece(current_state, first_action, current_player)
        current_player = -current_player
        
        # Simular algunos movimientos
        for _ in range(8):  # Profundidad limitada
            available_cols = [c for c in range(7) if current_state[0, c] == 0]
            if not available_cols:
                return 0.0
            
            # Acción aleatoria simple en simulación
            action = np.random.choice(available_cols)
            current_state = self._drop_piece(current_state, action, current_player)
            
            if self._check_win(current_state, current_player):
                return 1.0 if current_player == 1 else -1.0
            
            current_player = -current_player
        
        return 0.5  # Empate por defecto
    
    # Métodos existentes de Alex...
    def _would_win(self, state, col, player):
        test_state = self._drop_piece(state.copy(), col, player)
        return self._check_win(test_state, player)
    
    def _drop_piece(self, state, col, player):
        for row in range(5, -1, -1):
            if state[row, col] == 0:
                state[row, col] = player
                return state
        return state
    
    def _check_win(self, state, player):
        rows, cols = state.shape
        for row in range(rows):
            for col in range(cols - 3):
                if all(state[row, col + i] == player for i in range(4)):
                    return True
        for row in range(rows - 3):
            for col in range(cols):
                if all(state[row + i, col] == player for i in range(4)):
                    return True
        for row in range(rows - 3):
            for col in range(cols - 3):
                if all(state[row + i, col + i] == player for i in range(4)):
                    return True
        for row in range(3, rows):
            for col in range(cols - 3):
                if all(state[row - i, col + i] == player for i in range(4)):
                    return True
        return False