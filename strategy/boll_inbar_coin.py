from numba.experimental import jitclass

from .boll_inbar import (FCOLS, StrategyPosition, factor, get_default_factor_params_list,
                         get_default_strategy_params_list)


@jitclass
class Strategy:
    stra_pos: StrategyPosition
    face_value: float

    def __init__(self, leverage, face_value):
        self.stra_pos = StrategyPosition(leverage)
        self.face_value = face_value

    def on_bar(self, candle, factors, pos, equity):
        cl = candle['close']
        expo = equity + pos * self.face_value / cl
        equity_usd = equity * cl
        tar_expo = self.stra_pos.on_bar(candle, factors, expo, equity_usd)

        tar_pos = int((tar_expo - equity) * cl / self.face_value)
        return tar_pos
