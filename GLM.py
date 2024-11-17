from scipy.stats import norm, bernoulli ,poisson
import numpy as np
from scipy.optimize import minimize


class GLM:

    def __init__(self, x, y, start_b_value = 0.1):
        """
        Constructor method to create an instance of the GLM class.

        Parameters:
            x: x values.
            y: y value.
            start_b_value: the default beta value to be set.
        """
        
        self._x = x
        self._y = y

        #Dynamically get the number of params depending on input (x value columns).
        self._num_params = x.shape[1]

        #Set the given beta value by the user (0.1 as default) to be the current beta values.
        self._b_values = np.repeat(start_b_value, self._num_params)

    
    def fit(self):
        """
        Finds the optimal beta values and fit them to be the new beta values.

        Returns: _b_values: is set to be the new optimal beta values.
        """

        #Minimizing the _negativeLlik function to find optimal beta values
        results = minimize(self._negativeLlik, self._b_values, args =(self._x,self._y))

        #Set the new beta values
        self._b_values = results['x']
        return self._b_values 

    def predict(self,new_x):
        """
        Method to predict y values.

        Parameters: new_x: set of x values to predict y.

        Returns: mu: list of predictions for y.
        """

        eta = np.matmul(new_x,self._b_values)
        
        #Use the uniqueMu method within each subclass to create the specific mu calculation.
        mu = self.uniqueMu(eta)
        return mu 

    def _negativeLlik(self,b_values,x,y):
       """
       Finds the negative Log-likelihood so that we can use it in the minimizing function.

       Returns: negative of the uniqueLlik function.
       """

       return -self.uniqueLlik(b_values,x,y)


    #Abstract method to be implemented the specific behavior of llik function in each subclass.
    def uniqueLlik(self,b_values,x,y):
        raise NotImplementedError
    
    #Abstract method to implement the unique mu calculation in each subclass.
    def uniqueMu(self,eta):
       raise NotImplementedError
    

class ND(GLM):
    def __init__(self,x,y):
        """
        Constructor method to create an instance of the ND subclass.

        Parameters:
            x: x values.
            y: y value.
        """
    
        super().__init__(x,y)

    def uniqueLlik(self,b_values,x,y):
        """
        Finds the Log-likelihood for a Normal distribution.

        Returns: Log-likelihood for ND.
        """

        eta = np.matmul(x,b_values)
        mu = self.uniqueMu(eta)

        #Log-likelihood for a normal distribution
        llik = np.sum(norm.logpdf(y,mu))

        return llik
    
    def uniqueMu(self,eta):
       """Returns the unique mu calculation for a Normal distribution"""
       mu = eta
       return mu

       
class PD(GLM):
    def __init__(self,x,y):

        """
        Constructor method to create an instance of the PD subclass.

        Parameters:
            x: x values.
            y: y value.
        """

        super().__init__(x,y)

    def uniqueLlik(self,b_values,x,y):
        """
        Finds the Log-likelihood for a Poisson distribution.

        Returns: Log-likelihood for PD.
        """

        eta = np.matmul(x,b_values)
        mu = self.uniqueMu(eta)

        #Log-likelihood for a poisson distribution
        llik = np.sum(poisson.logpmf(y,mu))

        return llik
    
    def uniqueMu(self,eta):
        """Returns the unique mu calculation for a Poisson distribution."""
        mu = np.exp(eta)
        return mu

class BD(GLM):

    def __init__(self,x,y):
     super().__init__(x,y)

    def uniqueLlik(self,b_values,x,y):
        """
        Finds the Log-likelihood for a Bernoulli distribution.

        Returns: Log-likelihood for BD.
        """

        eta = np.matmul(x,b_values)
        mu = self.uniqueMu(eta)

        #Log-likelihood for a bernoulli distribution
        llik = np.sum(bernoulli.logpmf(y,mu))
        return llik
    
    def uniqueMu(self,eta):
        """Returns the unique mu calculation for a Bernoulli distribution."""
        mu = 1/(1+np.exp(-eta))
        return mu
    

