import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import optuna
from optuna.distributions import IntDistribution ,FloatDistribution, CategoricalDistribution
from optuna.integration import XGBoostPruningCallback, OptunaSearchCV 


class HyperXGBoost:
    
    def __init__(self, task:str = 'regression', model_params: dict = {}, n_jobs = 1):
         
        self._DEFAULT_TASKS = ['regression', 'classification']
        self._FS = (14, 6)  # figure size
        self._RS = 124  # random state
        self._N_JOBS = n_jobs  # number of parallel threads
        self._N_SPLITS = 10 # repeated K-folds
        self._N_REPEATS = 1 # repeated K-folds
        self._N_TRIALS = 100 # Optuna
        self._MULTIVARIATE = True # Optuna
        self._EARLY_STOPPING_ROUNDS = 100 # XGBoost
        self.model_params_ = model_params
        self.task_ = task if task in self._DEFAULT_TASKS else raise_error()
    
    def raise_error():
        raise ValueError(f"Mentioned {self.task_} is not supported !!! Choose either 'regression' or 'classification'.")
    
    def load_data(self, file_path:str, file_type:str = 'csv', **kwargs):
        #self.data_ = pd.read_csv(file_path, **kwargs)
        pass
    
    def train_test_split_Xy(self, X,y, test_size = 0.3, random_state = 42,**kwargs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, **kwargs)
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, X_test, y_train, y_test, study_n_trials, params:dict = {}):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        if self.task_ == 'regression':
            def objective(trial, verbose=1, n_jobs=self._N_JOBS):
                
                param = {
                    'max_depth': trial.suggest_int('max_depth', 5,20),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.01, 1.0),
                    'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
                    #'random_state': trial.suggest_categorical('random_state', [10,20,30,40,60,100])
                }
                
                if self.model_params_:
                    params.update(self.model_params_)
                #evalMetric = ''
                model = XGBRegressor(**param, early_stopping_rounds=100)
                #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_0-' + evalMetric)
                
                model.fit(self.X_train, self.y_train, eval_set = [(X_test, y_test)], verbose = False)
                y_pred = model.predict(self.X_test)
                return mean_squared_error(self.y_test, y_pred)
            
            study = optuna.create_study(direction='minimize', study_name='regression')
            study.optimize(objective, n_trials = study_n_trials)
            
            self.study_ = study
            self.best_params_ = study.best_params
            self.best_value_ = study.best_value
            self.best_trial_ = study.best_trial
            
            return self.study_
        
        elif self.task == 'classification':
            pass
        
        else:
            raise ImplementedError(f"""
                Mentioned {self.task_} is not supported !!! Choose either 'regression' or 'classification'.
            """)
                
    
    def predict(self, X_test = None, y_test = None, metrics_summary: bool = True):
        
        if (X_test and y_test) is not None:
            self.X_test = X_test
            self.y_test = y_test
        
        if self.task_ == 'regression':
            tuned_model = XGBRegressor(**self.best_params_)
        elif self.task_ == 'classification':
            pass
        else:
            raise_error()
        tuned_model.fit(self.X_train, self.y_train)
        self.y_pred = tuned_model.predict(self.X_test)

        if metrics_summary:
            if self.task_ == 'regression':
                print(f'MAE: {mean_absolute_error(self.y_test, self.y_pred)}')
                print(f'MSE: {mean_squared_error(self.y_test, self.y_pred)}')
                print(f'RMSE: {np.sqrt(mean_squared_error(self.y_test, self.y_pred))}')
                print(f'R2_Score: {r2_score(self.y_test, self.y_pred)}')
            
            elif self.task_ == 'classfication':
                pass
            
            else:
                raise_error()
        return self.y_pred
    
    def get_metrics_summary(self, as_dataframe: bool = True):
        if as_dataframe:
            metric_dict = {
                'MAE' : mean_absolute_error(self.y_test, self.y_pred),
                'MSE': mean_squared_error(self.y_test, self.y_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
                'R2_Score': r2_score(self.y_test, self.y_pred)
            }
            metrics_df = pd.DataFrame(metric_dict.items()).rename({0:'Accuracy_Metrics', 1: 'Score'}, axis = 1).set_index('Accuracy_Metrics')
            return metrics_df
        else:
            print(f'MAE: {mean_absolute_error(self.y_test, self.y_pred)}')
            print(f'MSE: {mean_squared_error(self.y_test, self.y_pred)}')
            print(f'RMSE: {np.sqrt(mean_squared_error(self.y_test, self.y_pred))}')
            print(f'R2_Score: {r2_score(self.y_test, self.y_pred)}')
            
    
    def save(self, model, file_path):
        output_file = open(f"{file_path}/{model}_{self.task_}.pickle", "w")
        pickle.dump(self,output_file)
    
    @classmethod
    def load(cls, model, file_path):
        input_file = open(f"{file_path}{model}_{self.task_}.pickle", "rb")
        return pickle.load(input_file)