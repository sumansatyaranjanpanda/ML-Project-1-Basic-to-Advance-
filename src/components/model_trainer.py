import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models: dict,params:dict):
        report = {}
        best_models = {}
        for name, model in models.items():
            logging.info(f"Training model: {name}")
            param_grid = params.get(name, {})

            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=0, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_ ##This gives you the actual model object (e.g., a trained RandomForestRegressor) with the best hyperparameters already set.
            else:
                model.fit(X_train, y_train)
                best_model = model


            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[name] = score
            best_models[name]=best_model  ### some hyperparameter tunned model and some other not hyperparameter tunned model.
            logging.info(f"{name} R2 Score: {score}")

        return report,best_models

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into features and target")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "BaggingRegressor": BaggingRegressor(),
                "SVR": SVR(),
                "KNeighbors": KNeighborsRegressor()
            }

            params = {
                "LinearRegression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.01, 0.1, 1.0]},
                "ElasticNet": {"alpha": [0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
                "DecisionTree": {"max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
                "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
                "BaggingRegressor": {"n_estimators": [10, 50]},
                "SVR": {"C": [0.1, 1.0], "kernel": ["linear", "rbf"]},
                "KNeighbors": {"n_neighbors": [3, 5, 7]}
            }

            model_report,best_models = self.evaluate_models(X_train, y_train, X_test, y_test, models,params)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_models[best_model_name]  ## already hyperparameter tunning model

            if best_model_score < 0.6:
                raise CustomException("No good model found (R2 < 0.6)", sys)

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return {
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                "model_path": self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:
            raise CustomException(e, sys)