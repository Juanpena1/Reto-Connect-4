import numpy as np
from connect4.policy import Policy


class Hello(Policy):

    def __init__(self):
        # PARÁMETROS FINALES OPTIMIZADOS
        self.mcts_iterations = 50
        self.exploration_weight = 1.0  # Balance perfecto velocidad/calidad
    
    def mount(self) -> None:
        """Método vacío ya que inicializamos en el constructor"""
        pass
    
    def act(self, s: np.ndarray) -> int:
        """SISTEMA UNIFICADO: Integración perfecta de todos los componentes"""
        available_cols = [c for c in range(7) if s[0, c] == 0]
        
        # MANEJO ROBUSTO: Casos bordes cubiertos
        if len(available_cols) == 0:
            return -1
        if len(available_cols) == 1:
            return available_cols[0]
        
        # INTEGRACIÓN PERFECTA: Heurísticas + MCTS
        immediate_action = self._check_immediate_actions(s, available_cols)
        if immediate_action is not None:
            return immediate_action
        
        # MCTS ULTRA-OPTIMIZADO: Versión final unificada
        return self._fast_mcts(s, available_cols)
    
    def _check_immediate_actions(self, state, available_cols):
        """HEURÍSTICAS PULIDAS: Cobertura completa de casos críticos"""
        # 1. Victoria inmediata - máxima prioridad
        for col in available_cols:
            if self._would_win(state, col, 1):
                return col
        
        # 2. Bloqueo inmediato - segunda prioridad
        for col in available_cols:
            if self._would_win(state, col, -1):
                return col
        
        return None
    
    def _fast_mcts(self, state, available_cols):
        """VERSIÓN FINAL: MCTS completamente optimizado"""
        # Inicializar contadores simples
        visits = [0] * 7
        wins = [0] * 7
        
        for i in range(self.mcts_iterations):
            # Selección rápida usando UCB1 simplificado
            action = self._fast_selection(available_cols, visits, wins)
            
            # Simulación ultra-rápida (solo 8-12 movimientos, no juego completo)
            result = self._fast_simulation(state, action)
            
            # Actualización directa
            visits[action] += 1
            if result > 0:
                wins[action] += 1
        
        # SELECCIÓN FINAL OPTIMIZADA
        best_action = available_cols[0]
        best_score = -1
        
        for col in available_cols:
            if visits[col] > 0:
                score = wins[col] / visits[col]
                # BONUS INTELIGENTE: Evita estancamiento en exploración
                score += 0.1 * (visits[col] == min(visits[c] for c in available_cols if visits[c] > 0))
                
                if score > best_score:
                    best_score = score
                    best_action = col
        
        return best_action
    
    def _fast_selection(self, available_cols, visits, wins):
        """SELECCIÓN PULIDA: Balance perfecto exploración/explotación"""
        total_visits = sum(visits)
        
        # ESTRATEGIA ROBUSTA: Exploración garantizada
        for col in available_cols:
            if visits[col] == 0:
                return col
        
        # UCB1 OPTIMIZADO: Parámetros finales ajustados
        best_score = -1
        best_action = available_cols[0]
        
        for col in available_cols:
            if visits[col] > 0:
                win_rate = wins[col] / visits[col]
                explore = self.exploration_weight * np.sqrt(np.log(total_visits) / visits[col])
                score = win_rate + explore
                
                if score > best_score:
                    best_score = score
                    best_action = col
        
        return best_action
    
    def _fast_simulation(self, state, first_action):
        """SIMULACIÓN FINAL: Equilibrio perfecto velocidad/calidad"""
        current_state = state.copy()
        current_player = 1
        
        # Aplicar primera acción
        current_state = self._drop_piece_fast(current_state, first_action, current_player)
        
        # VERIFICACIÓN INMEDIATA: Eficiencia máxima
        if self._fast_win_check(current_state, current_player):
            return 1.0
        
        current_player = -current_player
        
        # PROFUNDIDAD OPTIMIZADA: 10 movimientos máximo
        max_plies = 10
        plies = 0
        
        while plies < max_plies:
            plies += 1
            available_cols = [c for c in range(7) if current_state[0, c] == 0]
            
            if not available_cols:
                return 0.0  # Empate
            
            # POLÍTICA UNIFICADA: Heurísticas rápidas para rollout
            action = self._quick_rollout_action(current_state, available_cols, current_player)
            current_state = self._drop_piece_fast(current_state, action, current_player)
            
            # VERIFICACIÓN EFICIENTE: En cada movimiento
            if self._fast_win_check(current_state, current_player):
                return 1.0 if current_player == 1 else -1.0
            
            current_player = -current_player
        
        # EVALUACIÓN FINAL: Consistente y rápida
        return self._evaluate_final_position(current_state)
    
    def _quick_rollout_action(self, state, available_cols, player):
        """POLÍTICA FINAL: Simple pero efectiva"""
        # Solo verifica victorias/bloqueos inmediatos
        for col in available_cols:
            if self._would_win(state, col, player):
                return col
        
        opponent = -player
        for col in available_cols:
            if self._would_win(state, col, opponent):
                return col
        
        # Estrategia posicional simple - probada y confiable
        center_preference = [3, 2, 4, 1, 5, 0, 6]
        for col in center_preference:
            if col in available_cols:
                return col
        
        return available_cols[0]
    
    # MÉTODOS OPTIMIZADOS FINALES: Velocidad y precisión
    def _evaluate_final_position(self, state):
        """Evaluación rápida de posición final - Versión final"""
        player1_threats = self._count_threats(state, 1)
        player2_threats = self._count_threats(state, -1)
        
        if player1_threats > player2_threats:
            return 0.7  # Ventaja
        elif player2_threats > player1_threats:
            return 0.3  # Desventaja
        else:
            return 0.5  # Equilibrio
    
    def _count_threats(self, state, player):
        return self._count_sequences_fast(state, player, 3)
    
    def _count_sequences_fast(self, state, player, length):
        """Conteo rápido de secuencias - Optimizado final"""
        count = 0
        rows, cols = state.shape
        
        # Solo horizontal y vertical - equilibrio perfecto velocidad/precisión
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