import numpy as np
from connect4.policy import Policy


class Hello(Policy):

    def __init__(self):
        pass
    
    def mount(self) -> None:
        pass
    
    def act(self, s: np.ndarray) -> int:
        """Decisión basada en heurísticas básicas"""
        available_cols = [c for c in range(7) if s[0, c] == 0]
        
        if len(available_cols) == 0:
            return -1
        if len(available_cols) == 1:
            return available_cols[0]
        
        # 1. Victoria inmediata
        for col in available_cols:
            if self._would_win(s, col, 1):
                return col
        
        # 2. Bloqueo inmediato
        for col in available_cols:
            if self._would_win(s, col, -1):
                return col
        
        # 3. Estrategia posicional - centro primero
        center_cols = [3, 2, 4, 1, 5, 0, 6]
        for col in center_cols:
            if col in available_cols:
                return col
        
        return available_cols[0]
    
    def _would_win(self, state, col, player):
        """Verifica si esta movida resulta en victoria"""
        test_state = self._drop_piece(state.copy(), col, player)
        return self._check_win(test_state, player)
    
    def _drop_piece(self, state, col, player):
        """Coloca una ficha en la columna"""
        for row in range(5, -1, -1):
            if state[row, col] == 0:
                state[row, col] = player
                return state
        return state
    
    def _check_win(self, state, player):
        """Verificación completa de victoria"""
        rows, cols = state.shape
        
        # Horizontal
        for row in range(rows):
            for col in range(cols - 3):
                if all(state[row, col + i] == player for i in range(4)):
                    return True
        
        # Vertical
        for row in range(rows - 3):
            for col in range(cols):
                if all(state[row + i, col] == player for i in range(4)):
                    return True
        
        # Diagonal \
        for row in range(rows - 3):
            for col in range(cols - 3):
                if all(state[row + i, col + i] == player for i in range(4)):
                    return True
        
        # Diagonal /
        for row in range(3, rows):
            for col in range(cols - 3):
                if all(state[row - i, col + i] == player for i in range(4)):
                    return True
        
        return False