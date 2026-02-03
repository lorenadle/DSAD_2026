import numpy as np
import pandas as pd


def nan_replace(x:np.ndarray):
    is_nan = np.isnan(x)
    # print(is_nan)
    k = np.where(is_nan)
    # print(k)
    x[k] = np.nanmean(x[:,k[1]],axis=0)

def standardizare(x:np.ndarray,scal=True,ddof=0):
    x_ = x - np.mean(x,axis=0)
    if scal:
        x_ = x_ / np.std(x,axis=0,ddof=ddof)
    return x_

def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.to_csv(nume_fisier_output)

def calcul_corelatii_covariante(x:np.ndarray, y:np.ndarray):
    g = np.unique(y)
    corelatii_covariante = {}
    for v in g:
        x_ = x[y==v,:]
        r = np.round(np.corrcoef(x_,rowvar=False,ddof=0),3)
        v_g =np.round(np.cov(x_,rowvar=False,ddof=0),3)
        corelatii_covariante[v] = (r,v_g)
    return corelatii_covariante



