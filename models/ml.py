"""梯度提升回归器及验证集参数选择。"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid


def fit_ml_model(train_x, train_y, valid_x, valid_y):
    # 使用原有 GBDT 候选范围，以验证集 MSE 选择参数。
    model = GradientBoostingRegressor()
    parameter_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7]}

    # 每组参数仅使用训练集拟合，以验证集 MSE 选择后重新拟合。
    best_score = float('inf')
    for parameters in ParameterGrid(parameter_grid):
        model.set_params(**parameters)
        model.fit(train_x, train_y)
        prediction = model.predict(valid_x)
        score = mean_squared_error(valid_y, prediction)
        if score < best_score:
            best_score = score
            best_parameters = parameters
    model.set_params(**best_parameters)
    model.fit(train_x, train_y)
    return model, best_parameters, float(best_score)
