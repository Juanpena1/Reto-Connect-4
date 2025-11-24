import numpy as np
from connect4.policy import Policy


class Hello(Policy):

    def __init__(self):
        # PARÁMETROS OPTIMIZADOS: Más iteraciones, mejor balance
        self.mcts_iterations = 50  # Incrementado de 25
        self.exploration_weight = 1.0  # Ajustado para mejor convergencia
    
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
        
        return self._enhanced_mcts(s, available_cols)  # ¡MCTS mejorado!
    
    def _check_immediate_actions(self, state, available_cols):
        for col in available_cols:
            if self._would_win(state, col, 1):
                return col
        for col in available_cols:
            if self._would_win(state, col, -1):
                return col
        return None
    
    def _enhanced_mcts(self, state, available_cols):
        """MCTS MEJORADO: Con selección más inteligente"""
        visits = [0] * 7
        wins = [0] * 7
        
        for i in range(self.mcts_iterations):
            action = self._fast_selection(available_cols, visits, wins)
            result = self._fast_simulation(state, action)
            visits[action] += 1
            if result > 0:
                wins[action] += 1
        
        # SELECCIÓN MEJORADA: Considera ratio de victorias, no solo visitas
        best_action = available_cols[0]
        best_score = -1
        
        for col in available_cols:
            if visits[col] > 0:
                score = wins[col] / visits[col]
                # BONUS ANTI-STAGNATION: Premia acciones menos exploradas
                if visits[col] == min(visits[c] for c in available_cols if visits[c] > 0):
                    score += 0.1
                
                if score > best_score:
                    best_score = score
                    best_action = col
        
        return best_action
    
    def _fast_selection(self, available_cols, visits, wins):
        """SELECCIÓN OPTIMIZADA: Mejor balance exploración/explotación"""
        total_visits = sum(visits)
        
        # ESTRATEGIA MEJORADA: Exploración más agresiva
        unexplored = [col for col in available_cols if visits[col] == 0]
        if unexplored:
            # Priorizar columnas centrales no exploradas
            center_first = [3, 2, 4, 1, 5, 0, 6]
            for center_col in center_first:
                if center_col in unexplored:
                    return center_col
            return unexplored[0]
        
        # UCB1 con parámetros ajustados
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
    
    def _fast_simulation(self, state, first_action):
        """SIMULACIÓN MEJORADA: Con política de rollout más inteligente"""
        current_state = state.copy()
        current_player = 1
        
        current_state = self._drop_piece_fast(current_state, first_action, current_player)
        
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
            
            # POLÍTICA DE ROLLOUT MEJORADA: Más estratégica
            action = self._strategic_rollout_action(current_state, available_cols, current_player)
            current_state = self._drop_piece_fast(current_state, action, current_player)
            
            if self._fast_win_check(current_state, current_player):
                return 1.0 if current_player == 1 else -1.0
            
            current_player = -current_player
        
        return self._evaluate_final_position(current_state)
    
    def _strategic_rollout_action(self, state, available_cols, player):
        """POLÍTICA INTELIGENTE: Evalúa potencial ofensivo/defensivo"""
        # 1. Victoria inmediata
        for col in available_cols:
            if self._would_win(state, col, player):
                return col
        
        # 2. Bloqueo defensivo
        opponent = -player
        for col in available_cols:
            if self._would_win(state, col, opponent):
                return col
        
        # 3. Crear amenazas múltiples
        best_threat_col = None
        best_threat_score = -1
        
        for col in available_cols:
            threat_score = self._evaluate_threat_potential(state, col, player)
            if threat_score > best_threat_score:
                best_threat_score = threat_score
                best_threat_col = col
        
        if best_threat_score > 2:  # Umbral para amenazas significativas
            return best_threat_col
        
        # 4. Estrategia posicional
        center_preference = [3, 2, 4, 1, 5, 0, 6]
        for col in center_preference:
            if col in available_cols:
                return col
        
        return available_cols[0]
    
    def _evaluate_threat_potential(self, state, col, player):
        """EVALUACIÓN DE AMENAZAS: Cuántas oportunidades crea esta movida"""
        test_state = self._drop_piece_fast(state.copy(), col, player)
        score = 0
        
        # Bonus por secuencias de 3
        score += self._count_sequences_fast(test_state, player, 3) * 3
        
        # Bonus por secuencias de 2 que pueden convertirse en 3
        score += self._count_sequences_fast(test_state, player, 2) * 1
        
        return score

    def _evaluate_final_position(self, state):
        player1_threats = self._count_threats(state, 1)
        player2_threats = self._count_threats(state, -1)
        if player1_threats > player2_threats:
            return 0.7
        elif player2_threats > player1_threats:
            return 0.3
        else:
            return 0.5
    
    def _count_threats(self, state, player):
        return self._count_sequences_fast(state, player, 3)
    
    def _count_sequences_fast(self, state, player, length):
        count = 0
        rows, cols = state.shape
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
        return self._count_sequences_fast(state, player, 4) > 0
    
    def _would_win(self, state, col, player):
        test_state = self._drop_piece_fast(state.copy(), col, player)
        return self._fast_win_check(test_state, player)
    
    def _drop_piece_fast(self, state, col, player):
        for row in range(5, -1, -1):
            if state[row, col] == 0:
                state[row, col] = player
                return state
        return state
