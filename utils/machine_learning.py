import optuna, numpy as np, pickle
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


class MachineLearning:
    def __init__(self, model: str) -> None:
        """
        Class constructor
        
        ## Parameters
        `model` : str
            A model to use. One of:
            - linreg
            - knn
            - tree
            - rf
            - xgb
        """
        models = {
            'linreg': LinearRegression,
            'knn': KNeighborsRegressor,
            'tree': DecisionTreeRegressor,
            'rf': RandomForestRegressor,
            'xgb': GradientBoostingRegressor
        }
        self.model = models[model]
        self.model_name = model
        self.error_metric = mean_squared_log_error
        optuna.logging.set_verbosity(0)

    def fit(self, n_trials: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, *args) -> None:
        """
        Find optimal hyperparameters and fit the model

        ## Parameters
        `n_trials` : int
            Number of optimization iterations

        `X_train`, `y_train`, `X_test`, `y_test` : np.ndarray
            Training and test datasets
        
        `*args` : tuple
            Tuples containing hyperparameters and their values. If a hyperparameter is:
            - Numeric, then the tuple passed is (hyperparameter_name, min, max)
            - Categorical, then the tuple passed is (hyperparameter_name, [val_1, val_2, ..., val_n])
        """
        def objective(trial, *args):
            """
            Define objective for `optuna.study`
            """
            params = {}
            for arg in args:
                if len(arg) > 2:
                    if type(arg[1]) == int:
                        params[arg[0]] = trial.suggest_int(arg[0], arg[1], arg[2])
                    else:
                        params[arg[0]] = trial.suggest_float(arg[0], arg[1], arg[2])
                else:
                    params[arg[0]] = trial.suggest_categorical(arg[0], arg[1])

            model = self.model()
            model.set_params(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            error = self.error_metric(y_test, y_pred)
            return error

        study = optuna.create_study()
        study.optimize(lambda trial: objective(trial, *args), n_trials)

        model = self.model()
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)

        self.model = model

    def predict(self, data_set: np.ndarray) -> np.ndarray:
        """
        Generate predictions

        ## Parameters
        `data_set` : np.ndarray
            Either X_train, or X_test

        ## Returns
            Numpy array of predictions. Note the values are on a log scale
        """
        return self.model.predict(data_set)

    def get_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate root mean squared error (RMSE)

        ## Parameters
        `y_true` : np.ndarray
            Either y_train or y_test
        
        `y_pred` : np.ndarray
            Array of predictions with respect to `y_true`

        ## Returns
        `rmse` : float
            A value of RMSE (the closer to 0, the better)
        """
        return mean_squared_error(y_true, y_pred, squared=False)

    def save(self) -> None:
        """
        Save a model into a pickle file and place into "models" folder
        """
        with open(f'models/{self.model_name}.pkl', 'wb') as f:
            pickle.dump(self.model, f)