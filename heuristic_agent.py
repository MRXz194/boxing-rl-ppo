"""
Agent luật (rule-based) cho Boxing, dùng làm baseline. Không cần train.

Đọc vị trí 2 võ sĩ từ RAM rồi:
  - di chuyển để đứng ngang hàng (cùng y) với đối thủ,
  - giữ khoảng cách x vừa tầm đấm,
  - vào tầm thì đấm (FIRE).

Chạy đánh giá:  python evaluate.py --agent heuristic --episodes 30
"""
import numpy as np

from envs import read_state

# (dx, dy) -> action; y tăng là đi xuống
MOVE = {(0, 0): 0, (0, -1): 2, (1, 0): 3, (-1, 0): 4, (0, 1): 5,
        (1, -1): 6, (-1, -1): 7, (1, 1): 8, (-1, 1): 9}
MOVE_FIRE = {(0, 0): 1, (0, -1): 10, (1, 0): 11, (-1, 0): 12, (0, 1): 13,
             (1, -1): 14, (-1, -1): 15, (1, 1): 16, (-1, 1): 17}


class HeuristicAgent:
    def __init__(self, punch_dist_x: int = 28, y_tolerance: int = 10):
        self.punch_dist_x = punch_dist_x
        self.y_tolerance = y_tolerance

    def act(self, ram: np.ndarray) -> int:
        s = read_state(ram)
        px, py, ex, ey = s["player_x"], s["player_y"], s["enemy_x"], s["enemy_y"]

        # Trục y: tiến về ngang hàng với đối thủ
        dy = int(np.sign(ey - py)) if abs(ey - py) > self.y_tolerance else 0

        # Trục x: giữ khoảng cách vừa tầm
        side = 1 if ex >= px else -1
        gap = abs(ex - px)
        if gap > self.punch_dist_x + 2:
            dx = side
        elif gap < self.punch_dist_x - 6:
            dx = -side
        else:
            dx = 0

        in_range = abs(ey - py) <= self.y_tolerance and gap <= self.punch_dist_x + 4
        return (MOVE_FIRE if in_range else MOVE)[(dx, dy)]
