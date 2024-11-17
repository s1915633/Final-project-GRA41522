
from GLM import ND, PD, BD
from DataLoader import SMLoader, CSVLoader, WebLoader
import statsmodels.api as sm
import numpy as np
import argparse

#Create a parser object
parser = argparse.ArgumentParser(description="Lets the user run flexible unit tests on the classes: DataLoaders and GLMs.")

#add arguments that can be input by the user for flexible tests
parser.add_argument("--dataloader",required=True, choices=["SM","CSV","WEB"], help="Input the dataloader you want to test (SM, CSV, or WEB)", type=str)
parser.add_argument("--dset",required=True, help="Input the dataset you want to test. ", type=str)
parser.add_argument("--model",required=True,choices= ["ND","PD","BD"], help="Input the type of GLM you want to test. (ND, PD, BD) ", type=str)
parser.add_argument("--predictors",required=True, help="Input the set of x values you want use as predictors.", type=str)
parser.add_argument("--y",required=True, help="Input the column you want to predict (Y value)", type=str)

#Save the arguments as args
args = parser.parse_args()

def unitTests(args):

    #first we let the user choose the dataLoader type, then load the given dataset they input.
    if args.dataloader == "SM":
        loader = SMLoader(args.dset,"carData") #Here it is assumed that "carData actually is the library input"

    elif args.dataloader == "CSV":
        loader = CSVLoader(args.dset)

    elif args.dataloader == "WEB":
        loader = WebLoader(args.dset)

    #Load the dataset
    loader.loadData()

    #split the x columns on "," to input a list of columns 
    predictors = args.predictors.split(",")

    #set the x and y value
    loader.setX(predictors)
    loader.setY(args.y)

    #use get methods to get X and y
    X = loader.getX()
    y = loader.getY()

    #if the user inputs ND, we load the normal distribution class, and the SM class that is the same
    if args.model == "ND":
        model_distribution_GML = ND(X, y)
        model_distribution_SM = sm.GLM(y, X, family=sm.families.Gaussian())

    #if the user inputs PD, we load the Poisson distribution class, and the SM class that is the same
    elif args.model == "PD":
        model_distribution_GML = PD(X, y)
        model_distribution_SM = sm.GLM(y, X, family=sm.families.Poisson())

    #if the user inputs BD, we load the Poisson distribution class, and the SM class that is the same
    elif args.model == "BD":
        model_distribution_GML = BD(X, y)
        model_distribution_SM = sm.GLM(y, X, family=sm.families.Binomial())

    #get the beta values from GLM classes and SM
    fitted_model_distribution_GML = model_distribution_GML.fit()
    fitted_model_distribution_SM  = model_distribution_SM.fit()

    #get predictions for y for GLM classes and SM
    predictions_GML = model_distribution_GML.predict(X)
    predictions_sm = fitted_model_distribution_SM.predict(X)

    print("Predictions match (True/False): ",  np.allclose(predictions_GML, predictions_sm)) 
    print("Beta values match (True/False): ",  np.allclose(fitted_model_distribution_GML, fitted_model_distribution_SM.params)) 


unitTests(args)
