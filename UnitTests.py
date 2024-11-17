from GLM import ND,PD, BD
import statsmodels.api as sm
import pandas as pd
import numpy as np
from DataLoader import CSVLoader, WebLoader, SMLoader

#Task 1 sequential unit tests.
def unitTestsGLM():

    data_ND = sm.datasets.get_rdataset("Duncan","carData").data
    data_BD = sm.datasets.spector.load_pandas().data
    data_PD = pd.read_csv("warpbreaks.csv")

    x_nd = data_ND[['income', 'education']].values 
    y_nd = data_ND['prestige'].values 

    x_bd = data_BD[['GPA', 'TUCE', 'PSI']].values 
    y_bd = data_BD['GRADE'].values 

    x_pd = data_PD[['wool', 'tension']]
    y_pd = data_PD['breaks']

    #Normal mod GML TEST
    normal_dist = ND(x_nd,y_nd)
    normal_dist.fit()
    predictions_nd = normal_dist.predict(x_nd)


    #Normal mod TEST
    normal_mod = sm.GLM(y_nd,x_nd,family = sm.families.Gaussian())
    normal_mod_fitted = normal_mod.fit()
    predictions_sm_nd = normal_mod_fitted.predict(x_nd)


    #bernoulli mod TEST GML
    bernoulli_dist = BD(x_bd,y_bd)
    bernoulli_dist.fit()
    predictions_bd = bernoulli_dist.predict(x_bd)


    #bernoulli mod TEST SM
    bernoulli_mod = sm.GLM(y_bd,x_bd,family = sm.families.Binomial())
    bernoulli_mod_fitted = bernoulli_mod.fit()
    predictions_sm_bernoulli = bernoulli_mod_fitted.predict(x_bd)


    #possion mod TEST GML
    possion_dist = PD(x_pd,y_pd)
    possion_dist.fit()
    predictions_pd = possion_dist.predict(x_pd)


    #Possion mod TEST SM
    possion_mod = sm.GLM(y_pd,x_pd,family = sm.families.Poisson())
    possion_mod_mod_fitted = possion_mod.fit()
    predictions_sm_possion = possion_mod_mod_fitted.predict(x_pd)


    #ND model predictions comparison
    if np.allclose(predictions_nd, predictions_sm_nd):
        print("Model predictions match for ND.")
    else:
        print("Error not a match for ND.")

    #PD model predictions comparison
    if np.allclose(predictions_pd, predictions_sm_possion):
        print("Model predictions match for PD.")
    else:
        print("Error not a match for PD.")

    #BD model predictions comparison
    if np.allclose(predictions_bd, predictions_sm_bernoulli):
        print("Model predictions match for BD.")
    else:
        print("Error not a match for BD.")


#Task 2 sequential unit tests.
def unitTestsLoader():

    #use the Dataloader class
    nd_loader = SMLoader("Duncan", "carData")
    nd_loader.loadData()
    nd_loader.setX(["income", "education"])
    nd_loader.setY('prestige')

    X_nd = nd_loader.getX()
    Y_nd = nd_loader.getY()

    normal_dist = ND(X_nd, Y_nd)
    normal_dist_fitted = normal_dist.fit()
    predictions_nd = normal_dist.predict(X_nd)
    normal_mod = sm.GLM(Y_nd, X_nd, family=sm.families.Gaussian())
    normal_mod_fitted = normal_mod.fit()
    predictions_sm_nd = normal_mod_fitted.predict(X_nd)

    print("Predictions match? (ND) ",  np.allclose(predictions_nd, predictions_sm_nd)) 
    print("Beta values match? (ND) ",  np.allclose(normal_dist_fitted, normal_mod_fitted.params)) 

    pd_loader = WebLoader ("https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv")
    pd_loader.loadData()
    pd_loader.setY("breaks")
    pd_loader.setX(["wool", "tension"])

    Y_pd = pd_loader.getY()
    X_pd = pd_loader.getX()

    #possion mod TEST GML
    possion_dist = PD(X_pd ,Y_pd)
    possion_dist_fitted = possion_dist.fit()
    predictions_pd = possion_dist.predict(X_pd)

    #Possion mod TEST SM
    possion_mod = sm.GLM(Y_pd,X_pd,family = sm.families.Poisson())
    possion_mod_fitted = possion_mod.fit()
    predictions_sm_possion = possion_mod_fitted.predict(X_pd)

    print("Beta values match? (PD) ",  np.allclose(possion_dist_fitted, possion_mod_fitted.params)) 
    print("Predictions match? (PD) ",  np.allclose(predictions_pd, predictions_sm_possion)) 

    bd_loader = CSVLoader("spector_dataset.csv")
    bd_loader.loadData()
    bd_loader.setX(['GPA', 'TUCE', 'PSI'])
    bd_loader.setY('GRADE')

    X_bd = bd_loader.getX()
    Y_bd = bd_loader.getY()

    bd_model = BD(X_bd, Y_bd)
    bd_model_fitted = bd_model.fit()
    predictions_bd = bd_model.predict(X_bd)

    #bernoulli mod TEST SM
    bernoulli_mod = sm.GLM(Y_bd,X_bd,family = sm.families.Binomial())
    bernoulli_mod_fitted = bernoulli_mod.fit()
    predictions_sm_bernoulli = bernoulli_mod_fitted.predict(X_bd)


    print("Beta values match? (BD) ",  np.allclose(bd_model_fitted, bernoulli_mod_fitted.params)) 
    print("Predictions match? (BD) ",  np.allclose(predictions_bd, predictions_sm_bernoulli)) 


unitTestsGLM()
unitTestsLoader()

