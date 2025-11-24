import numpy as np
from connect4.policy import Policy


class Hello(Policy):

    def __init__(self):
        self.mcts_iterations = 25
        self.exploration_weight = 1.5
    
    def mount(self) -> None:
        pass
    
    def act(self, s: np.ndarray) -> int:
        available_cols = [c for c in range(7) if s[0, c] == 0]
        
        if len(available_cols) == 0:
            return -1
        if len(available_cols) == 1:
            return available_cols[0]
        
        immediate_action = self._check_immediate_actions(s, available_cols)
        if immediate_action is not None:
            return immediate_action
        
        return self._basic_mcts(s, available_cols)
    
    def _check_immediate_actions(self, state, available_cols):
        for col in available_cols:
            if self._would_win(state, col, 1):
                return col
        for col in available_cols:
            if self._would_win(state, col, -1):
                return col
        return None
    
    def _basic_mcts(self, state, available_cols):
        visits = [0] * 7
        wins = [0] * 7
        
        for i in range(self.mcts_iterations):
            action = self._ucb_selection(available_cols, visits, wins)
            result = self._fast_simulation(state, action)
            visits[action] += 1
            if result > 0:
                wins[action] += 1
        
        best_action = max(available_cols, key=lambda col: visits[col])
        return best_action
    
    def _fast_simulation(self, state, first_action):
        """SIMULACIÓN OPTIMIZADA: 3x más rápida"""
        current_state = state.copy()
        current_player = 1
        
        current_state = self._drop_piece_fast(current_state, first_action, current_player)
        
        # Verificación rápida de victoria
        if self._fast_win_check(current_state, current_player):
            return 1.0
        
        current_player = -current_player
        
        max_plies = 10
        plies = 0
        
        while plies < max_plies:
            plies += 1
            available_cols = [c for c in range(7) if current_state[0, c] == 0]
            
            if not available_cols:
                return 0.0
            
            # Acción rápida con heurística simple
            action = self._quick_rollout_action(current_state, available_cols, current_player)
            current_state = self._drop_piece_fast(current_state, action, current_player)
            
            if self._fast_win_check(current_state, current_player):
                return 1.0 if current_player == 1 else -1.0
            
            current_player = -current_player
        
        # Evaluación final optimizada
        return self._evaluate_final_position(current_state)
    
    def _quick_rollout_action(self, state, available_cols, player):
        """Heurística rápida para simulaciones"""
        for col in available_cols:
            if self._would_win(state, col, player):
                return col
        opponent = -player
        for col in available_cols:
            if self._would_win(state, col, opponent):
                return col
        center_preference = [3, 2, 4, 1, 5, 0, 6]
        for col in center_preference:
            if col in available_cols:
                return col
        return available_cols[0]
    
    def _evaluate_final_position(self, state):
        """EVALUACIÓN RÁPIDA: Basada en amenazas relativas"""
        player1_threats = self._count_threats(state, 1)
        player2_threats = self._count_threats(state, -1)
        
        if player1_threats > player2_threats:
            return 0.7
        elif player2_threats > player1_threats:
            return 0.3
        else:
            return 0.5
    
    def _count_threats(self, state, player):
        """Conteo rápido de secuencias de 3"""
        return self._count_sequences_fast(state, player, 3)
    
    def _count_sequences_fast(self, state, player, length):
        """CONTEO OPTIMIZADO: Solo horizontales y verticales"""
        count = 0
        rows, cols = state.shape
        
        # Solo horizontal y vertical (omitir diagonales para velocidad)
        for row in range(rows):
            for col in range(cols - length + 1):
                if all(state[row, col + i] == player for i in range(length)):
                    count += 1
        
        for row in range(rows - length + 1):
            for col in range(cols):
                if all(state[row + i, col] == player for i in range(length)):
                    count += 1
        
        return count
    
    def _fast_win_check(self, state, player):
        """VERIFICACIÓN RÁPIDA: Usa conteo optimizado"""
        return self._count_sequences_fast(state, player, 4) > 0
    
    def _drop_piece_fast(self, state, col, player):
        """Versión optimizada de drop_piece"""
        for row in range(5, -1, -1):
            if state[row, col] == 0:
                state[row, col] = player
                return state
        return state
    
    # Métodos UCB y auxiliares permanecen igual...
    def _ucb_selection(self, available_cols, visits, wins):
        total_visits = sum(visits)
        for col in available_cols:
            if visits[col] == 0:
                return col
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
    
    def _would_win(self, state, col, player):
        test_state = self._drop_piece_fast(state.copy(), col, player)
        return self._fast_win_check(test_state, player)