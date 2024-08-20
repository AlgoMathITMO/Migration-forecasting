from pandas import json_normalize
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import scipy.stats as sts
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_curve
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")



def plot_hist(data: pd.DataFrame, synth_data: pd.DataFrame, bins=40):
    for i in range(len(data.columns)):
        plt.hist(data.iloc[:, i], bins=bins, alpha=0.7, label='Real', color='black', density=True)
        plt.hist(synth_data.iloc[:, i], bins=bins, alpha=0.7, label='Synth', color='tab:red', density=True)

        plt.legend()
        plt.show()
        
        
        
def plot_qq(data_test, prediction):
    plt.plot(data_test['saldo'], data_test['saldo'], color='black')
    plt.scatter(data_test['saldo'], prediction )

    plt.xlabel('True')
    plt.ylabel('Prediction')

    plt.show()
    
    
# def plot_qq_with_axis(data_test, prediction, save=False):
#     plt.plot(data_test['saldo'], data_test['saldo'], color='tab:red', zorder=2, label='Identity line')
#     plt.scatter(data_test['saldo'], prediction ,zorder=3, color='tab:blue')

#     plt.xlabel('True')
#     plt.ylabel('Prediction')
    
#     plt.axhline(0., c='k', linewidth=0.7, zorder=1)
#     plt.axvline(0., c='k', linewidth=0.7, zorder=1)

#     plt.legend()
    
#     plt.tight_layout(pad=0.5)
    
#     if save:
#         plt.savefig('images/qq_plot.pdf', dpi=300)
        
#     plt.show()
    

def plot_qq_with_axis(data_test, prediction, save=False, norm=26466):
    plt.plot(data_test['saldo']*norm, data_test['saldo']*norm, color='tab:red', zorder=2, label='Identity line')
    plt.scatter(data_test['saldo']*norm, prediction*norm ,zorder=3, color='tab:blue')
    
    #Confidence interval
    lr = LinearRegression()
    lr.fit(data_test['saldo'].values.reshape(-1, 1)*norm, prediction.reshape(-1, 1)*norm)
    plt.plot(data_test['saldo']*norm, lr.intercept_ + lr.coef_[0] * data_test['saldo']*norm, zorder=1, color='green', label='OLS', alpha=0.8)
    
    t_crit = sts.t.ppf(1 - 0.05/2, df=len(prediction) - 2)
    x = sorted(data_test['saldo']*norm)
    # sigma = np.std(prediction*norm)
    sigma = np.sqrt(sum((prediction*norm - data_test['saldo']*norm)**2) / (len(prediction)-2))
    x_bar = np.mean(x)
    plt.fill_between(x, lr.intercept_ + lr.coef_[0] * x - t_crit * sigma * np.sqrt(  1/len(x) + (x - x_bar)**2/((x-x_bar)**2).sum()),
                 lr.intercept_ + lr.coef_[0] * x + t_crit * sigma * np.sqrt(  1/len(x) + (x - x_bar)**2 / ((x-x_bar)**2).sum()), 
                 color = 'green', alpha = 0.1, label = '95% CI', linestyle='-')


    plt.xlabel('True')
    plt.ylabel('Prediction')
    
    plt.axhline(0., c='k', linewidth=0.7, zorder=1)
    plt.axvline(0., c='k', linewidth=0.7, zorder=1)

    plt.legend()
    
    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig('images/qq_plot.pdf', dpi=300)
        
    plt.show()
    

def plot_abs(data_test, prediction):
    df_pred = pd.read_excel('Data/smallcitiesNY.xlsx').drop(30)
    plt.scatter(df_pred['popsize'], df_pred['abs'], label='Old', color='black')
    plt.scatter(df_pred['popsize'], np.abs(data_test.saldo.values - prediction) * 26466, label='New', color='red')

    plt.legend()
    plt.xlabel('Population (x10^3)')
    plt.ylabel('Abs. error (humans)')
    plt.show()
    
    
    
def plot_hist_kde(data1, data2):
    for col in data1.columns:
        data1[col].hist(bins=40, density=True, label='Data 1', color='tab:blue', alpha=0.7)
        data2[col].hist(bins=40, density=True, label='Data 2', color='tab:red', alpha=0.7)
        
        
        data1[col].plot.kde(label='Data 1 KDE', color='blue')
        data2[col].plot.kde(label='Data 2 KDE', color='black')
        
        plt.title(f'{col}')
        plt.legend()

        plt.show()
        
        
def plot_losses(losses):
    plt.plot(losses['gen_loss'], label='generator')
    plt.plot(losses['dis_loss'], label='discriminator')

    plt.legend()
    plt.show()
    

def plot_corr(data: pd.DataFrame, path: str = 'corr', save=False):
    plt.subplots(figsize=(15,15), dpi=100)
    
    sns.heatmap(np.around(data.corr(), 2), square=True, annot=True, annot_kws={"size":18}, cbar_kws={"shrink": 0.5})
    
    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig(path, dpi=300)
        
    plt.show()
    
    
# def plot_significance_ks(data_clf, synth_data_clf, data_reg, synth_data_reg, path='pvalue', save=False):
#     plt.bar(np.arange(len(data_clf.columns)), ks_test(data_clf, synth_data_clf.sample(len(data_clf))), color='tab:blue', alpha=0.7, label='classification')
#     plt.bar(np.arange(len(data_reg.columns)), ks_test(data_reg, synth_data_reg.sample(len(data_reg))), color='tab:red', alpha=0.7, label='regression')

#     plt.plot([-1, 14], [0.05, 0.05], color='black', label='significance level')

#     plt.xlabel('column index')
#     plt.xticks(ticks=np.arange(len(data_clf.columns)))
#     plt.ylabel('p-value')
#     plt.legend()

#     plt.tight_layout(pad=0.5)
#     if save:
#         plt.savefig(path, dpi=300)

#     plt.show()    
    
    
def plot_significance_ks(data_clf, synth_data_clf, data_reg, synth_data_reg, path='pvalue', save=False):
    plt.subplots(figsize=(8, 7))
    plt.bar(np.arange(len(data_clf.columns)), ks_test(data_clf, synth_data_clf.sample(len(data_clf))), color='tab:blue', alpha=0.7, label='classification')
    plt.bar(np.arange(len(data_reg.columns)), ks_test(data_reg, synth_data_reg.sample(len(data_reg))), color='tab:red', alpha=0.7, label='regression')

    plt.plot([-1, 14], [0.05, 0.05], color='black', label='significance level')

    # plt.xlabel('column index')
    # plt.xticks(ticks=np.arange(len(data_clf.columns)))
    plt.xticks(ticks=np.arange(len(data_clf.columns)), labels=list(data_reg.columns), rotation=90)
    plt.ylabel('p-value')
    plt.legend()

    plt.tight_layout(pad=0.5)
    if save:
        plt.savefig(path, dpi=300)

    plt.show()  
    
    
def plot_shap(model, data, synth_data, path='shap', save=False):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(synth_data[data.columns].drop('saldo', axis=1))

    shap.summary_plot(shap_values, synth_data[data.columns].drop('saldo', axis=1).astype("float"), show=False)

    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig(path, dpi=300)
        
    plt.show()
    
def plot_roc_curve(model, synth_data, path='roc_curve', save=False):
    fpr, tpr, thresholds = roc_curve(synth_data['saldo'], model.predict_proba(synth_data.drop('saldo', axis=1))[:, 1])

    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0., 1.], [0., 1.], color='black', label='Random guess')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()

    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig(path, dpi=300)
        
    plt.show()

    
def feature_importance(X: pd.DataFrame, y, path: str = 'fi', save=False):
    
    xgb = XGBRegressor()
    xgb.fit(X, y)


    explainer = shap.Explainer(xgb, X)
    shap_values = explainer(X)


    fi = np.array(sorted(list(zip(X.columns, np.abs(shap_values.values).mean(axis=0))), key=lambda x: x[1]))


    plt.subplots(figsize=(10,5), dpi=100)

    plt.barh(fi[:, 0], fi[:, 1].astype(float), color='tab:red')
    plt.xlabel('mean(|SHAP value|)')

    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig(path, dpi=300)

    plt.show()
    
    
def c2st(data: pd.DataFrame, synth_data: pd.DataFrame):
    
    statistic_arr = []

    for _ in range(0, 1000):
        synth_data = synth_data.sample(len(data))
        label = []

        for _ in range(len(data)):
            label.append(1)
            
        for _ in range(len(synth_data)):
            label.append(0)

        realFakeFeatures = np.vstack((data, synth_data))
        
        
        X_train, X_test, y_train, y_test = train_test_split(realFakeFeatures, label, test_size=0.3, shuffle=True, random_state=55555)
    
    
        clf = LogisticRegression().fit(X_train, y_train)
        predicted_y = clf.predict_proba(X_test)
        

        statistic = sum((predicted_y[:, 1] > 0.5).astype(int) == y_test) / len(y_test)
        statistic_arr.append(statistic)
        
    pvalue = 1 - sts.norm.cdf(np.mean(statistic_arr), loc=0.5, scale=np.sqrt(1/(4*len(y_test))))
    
    # p_bar = np.mean(predicted_y[:, 1])
    # pvalue = sts.binom.cdf(len(y_test) * (np.mean(statistic_arr)), n=len(synth_data), p=p_bar)
        
    return np.mean(statistic_arr), pvalue


def ks_test(data: pd.DataFrame, synth_data: pd.DataFrame):
    arr = []
    for col in data.columns:
        arr.append(sts.ks_2samp(data[col], synth_data[col]))
        
    arr = np.array(arr)
    
    return arr[:, 1]


