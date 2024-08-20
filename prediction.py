from sklearn.metrics import mean_absolute_percentage_error, make_scorer, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from xgboost import XGBClassifier, XGBRegressor
import optuna
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm


CV_FOLDS = 10


###########################################################################################
# CLASSIFICATION TASK
###########################################################################################

def objective_class(trial, datasetin, datasetout):
    params = {
        "objective": "binary:logistic",
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        # "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
    }

    model = XGBClassifier(**params)
    f1 = np.mean(cross_val_score(estimator=model, X=datasetin, y=datasetout, scoring="f1", cv=5))

    return f1



def classification(data_train, data_test, n_trials: int = 30, test_size: float = 0.2):

        # test data partition
        villagein = np.array(data_test[data_test.columns.drop('saldo')])
        villageout = np.array(data_test[['saldo']])
        
        # remove unnecessary columns
        data_train = data_train.drop(np.setdiff1d(data_train.columns, data_test.columns), axis=1)
        
        # train data partition
        datasetin = np.array(data_train[data_train.columns.drop('saldo')])
        datasetout = np.array(data_train[['saldo']])

        # train validation partition
        trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=test_size, random_state=42, shuffle=True)

        # Bayes hyperparameter tuning with cross-validation
        study = optuna.create_study(direction='maximize')
        study.optimize(partial(objective_class, datasetin=datasetin, datasetout=datasetout), n_trials=n_trials, show_progress_bar=False)

        # train model
        model = XGBClassifier(**study.best_params, random_state=0, n_jobs=-1)
        model.fit(trainin, trainout.ravel())
        
        # compute cross-validation score
        cv_score = np.mean(cross_val_score(estimator=model, X=testin, y=testout, scoring="f1", cv=CV_FOLDS))
        cv_roc_auc_score = np.mean(cross_val_score(estimator=model, X=testin, y=testout, scoring="roc_auc", cv=CV_FOLDS))

        # predictions
        predtrain = model.predict(trainin)
        errortrain = f1_score(trainout, predtrain)

        predval = model.predict(testin)
        errorval = f1_score(testout, predval)

        prediction = model.predict(villagein)
        errortest = f1_score(villageout, prediction)
        
        errors = {'F1 train': errortrain, 'F1 val': errorval, 'F1 real test': errortest, 'F1 CV': cv_score, 'roc_auc_score': cv_roc_auc_score}
        
        return model, errors
    
###########################################################################################




###########################################################################################
# REGRESSION TASK
###########################################################################################

mse_score = make_scorer(mean_squared_error, 
                        greater_is_better=False
                             )

mae_score = make_scorer(lambda x, y: mean_absolute_error(x, y) * 26466, 
                        greater_is_better=False
                             )

def objective_reg(trial, datasetin, datasetout):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 4)
    }

    model = XGBRegressor(**params)
    mse = np.mean(cross_val_score(estimator=model, X=datasetin, y=datasetout, scoring=mse_score, cv=5))
    # xtrain, xtest, ytrain, ytest = train_test_split(datasetin, datasetout, test_size=0.2, random_state=55555, shuffle=False)
    # model.fit(xtrain, ytrain) 
    # mse = mean_squared_error(ytest, model.predict(xtest))
    
    return mse


def regression(data_train, data_test, n_trials: int = 30, test_size: float = 0.2):
        # test data partition
        villagein = np.array(data_test[data_test.columns.drop('saldo')])
        villageout = np.array(data_test[['saldo']])
        
        # remove unnecessary columns
        data_train = data_train.drop(np.setdiff1d(data_train.columns, data_test.columns), axis=1)
        
        # train data partition
        datasetin = np.array(data_train[data_train.columns.drop('saldo')])
        datasetout = np.array(data_train[['saldo']])

        # train validation partition
        trainin, testin, trainout, testout = train_test_split(datasetin, datasetout, test_size=test_size, random_state=42, shuffle=True)

        # Bayes hyperparameter tuning with cross-validation
        study = optuna.create_study(direction='minimize')
        study.optimize(partial(objective_reg, datasetin=datasetin, datasetout=datasetout), n_trials=n_trials, show_progress_bar=False)

        # train model
        model = XGBRegressor(**study.best_params, n_jobs=-1)
        model.fit(trainin, trainout.ravel())
        
        cv_score = np.mean(cross_val_score(estimator=model, X=testin, y=testout, scoring=mae_score, cv=CV_FOLDS))
        
        predtrain = model.predict(trainin)
        errortrain = mean_absolute_percentage_error(trainout, predtrain)

        predval = model.predict(testin)
        errorval = mean_absolute_percentage_error(testout, predval)

        prediction = model.predict(villagein)
        errortest = mean_absolute_percentage_error(villageout, prediction)
        mse = mean_squared_error(villageout, prediction) * 10000
        
        errors = {'MAPE train': errortrain, 'MAPE val': errorval, 'MAPE real test': errortest, 'MAE CV': -cv_score, 'MSE x 10e-4': mse}
        
        return model, errors


###########################################################################################
    
    
    
    
    
###########################################################################################
# FINAL PREDICTION WITH CORRECTION
###########################################################################################

def correct_prediction(model_clf, model_reg, data_test):
    pred_class = model_clf.predict(data_test.drop('saldo', axis=1))
    prediction = np.abs(model_reg.predict(data_test.drop('saldo', axis=1)))
    prediction[np.where(pred_class == 0)[0]] = prediction[np.where(pred_class == 0)[0]] * (-1)
    
    return prediction


###########################################################################################



def train_models(data_reg: pd.DataFrame, data_clf: pd.DataFrame, data_big_reg, synth_data_reg, synth_data_clf, reg_model, clf_model, iter_num: int = 100):
    data_X_test = data_reg.drop('saldo', axis=1)
    data_y_test = data_reg['saldo'].values

    errors_mae_real_small = []
    errors_mse_real_small = []
    errors_mae_real_big = []
    errors_mse_real_big = []
    errors_mae = []
    errors_mse = []
    errors_mae_adj = []
    errors_mse_adj = []
    errors_f1 = []

    iters = tqdm(range(iter_num))
    
    for _ in iters:
        #real small cities
        data_reg = data_reg.sample(frac=1)
        X_train, X_test, y_train, y_test = train_test_split(data_reg.drop('saldo', axis=1), data_reg['saldo'], test_size=0.2, shuffle=False)
        
        model_reg = reg_model
        model_reg.fit(X_train, y_train)
        
        prediction = model_reg.predict(X_test)
        errors_mae_real_small.append(mean_absolute_error(y_test.values, prediction) * 26466)
        errors_mse_real_small.append(mean_squared_error(y_test.values, prediction) * 10000)
        
        
        #real big cities
        data_big_reg = data_big_reg.sample(frac=1)
        X_train, X_test, y_train, y_test = train_test_split(data_big_reg[data_reg.columns].drop('saldo', axis=1),
                                                            data_big_reg[data_reg.columns]['saldo'], test_size=0.2, shuffle=False)
        
        model_reg = reg_model
        model_reg.fit(X_train, y_train)
        
        prediction = model_reg.predict(data_X_test)
        errors_mae_real_big.append(mean_absolute_error(data_y_test, prediction) * 26466)
        errors_mse_real_big.append(mean_squared_error(data_y_test, prediction) * 10000)  
        
        
        #synth regression
        synth_data_reg = synth_data_reg.sample(frac=1)
        X_train, X_test, y_train, y_test = train_test_split(synth_data_reg.drop('saldo', axis=1), synth_data_reg['saldo'], test_size=0.2, shuffle=False)
        
        model_reg = reg_model
        model_reg.fit(X_train, y_train)
        
        prediction = model_reg.predict(data_X_test)
        errors_mae.append(mean_absolute_error(data_y_test, model_reg.predict(data_X_test)) * 26466)
        errors_mse.append(mean_squared_error(data_y_test, prediction) * 10000)
        
        
        #synth regression + synth classifier
        synth_data_reg = synth_data_reg.sample(frac=1)
        X_train, X_test, y_train, y_test = train_test_split(synth_data_reg.drop('saldo', axis=1), synth_data_reg['saldo'], test_size=0.2, shuffle=False)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(synth_data_clf.loc[synth_data_reg.index].drop('saldo', axis=1),
                                                                            synth_data_clf.loc[synth_data_reg.index]['saldo'], test_size=0.2, shuffle=False)
        
        model_clf = clf_model
        model_reg = reg_model
        
        model_reg.fit(X_train, y_train)
        model_clf.fit(X_train_clf, y_train_clf)
        
        
        pred_class = model_clf.predict(data_X_test)
        errors_f1.append(f1_score(data_clf['saldo'].values, pred_class))
        prediction_adj = np.abs(model_reg.predict(data_X_test))
        prediction_adj[np.where(pred_class == 0)[0]] = prediction_adj[np.where(pred_class == 0)[0]] * (-1)
        
        errors_mae_adj.append(mean_absolute_error(data_y_test, prediction_adj) * 26466)
        errors_mse_adj.append(mean_squared_error(data_y_test, prediction_adj) * 10000)
        
    errors = {'MAE_small': errors_mae_real_small, 'MSE_small': errors_mse_real_small,
              'MAE_big': errors_mae_real_big, 'MSE_big': errors_mse_real_big,
              'MAE_synth_reg': errors_mae, 'MSE_synth_reg': errors_mse,
              'MAE_adjusted': errors_mae_adj, 'MSE_adjusted': errors_mse_adj}
        
    return errors