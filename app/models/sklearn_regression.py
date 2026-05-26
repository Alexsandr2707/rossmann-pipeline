from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from app.models.base import PartialFitUpdateMixin, RefitUpdateMixin


class RefitDecisionTreeRegressor(RefitUpdateMixin, DecisionTreeRegressor):
    pass


class RefitKNeighborsRegressor(RefitUpdateMixin, KNeighborsRegressor):
    pass


class RefitRandomForestRegressor(RefitUpdateMixin, RandomForestRegressor):
    pass


class RefitRidgeRegressor(RefitUpdateMixin, Ridge):
    pass


class PartialFitSGDRegressor(PartialFitUpdateMixin, SGDRegressor):
    pass
