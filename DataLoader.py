import statsmodels.api as sm
import csv
import pandas as pd
import numpy as np


def requireDataset(func):
     """
     Decorator to ensure that a dataset is actually loaded, before the user calls methods that requires it.

     Parameters:
        func: The function the wrapper should add extended functionality to.
     """

     def wrapper(self,*args):
         #check if the dataset is None
         if self._dataset is None:
            raise ValueError("You must load a dataset before calling this method!")
         return func(self,*args)
     
     return wrapper

class DataLoader:
    """Constructor for the DataLoader superclass."""

    def __init__(self):

     self._dataset = None
     self._x_values = None
     self._y_value = None
        
    @requireDataset
    def setX(self, columns = None):
        """
        Sets the x values to be the X matrix. If the user doesn't input any columns, the default will every column excluding the last.

        Parameters:
            columns: the set of columns to be set as x values.
        """

        #we assume that the x values are all columns except the last one, if none was selected
        if columns is None:
            self._x_values = self._dataset[self._dataset.columns[:-1]]
        else:
            self._x_values = self._dataset[columns]

    @requireDataset
    def setY(self, column = None):
        """
        Sets the input of column to be Y. If no column is set, the default will be the last column.

        Parameters:
            column: the column that should be set as Y.
        """

        #we assume that the y value is the last value in the column
        if column is None:
            self._y_value = self._dataset[self._dataset.columns[-1]]
        else:
             self._y_value = self._dataset[column]

    @requireDataset
    def getX(self):
        """
        Getter method to get x values.
        Returns: the X matrix constisting of x columns.
        """

        return self._x_values.values

    @requireDataset
    def getY(self):
        """
        Getter method to get 
        Returns: Y column consisting of y values
        """

        return self._y_value.values
    
    @requireDataset
    def setNewY(self,column):
        """
        Lets a user set a new Y value (column) in the case that the last column should not be Y.
        Also updates the instance variables to prevent duplicate values.
        """

        #set the current Y, to be a part of x values again
        self._x_values[self._y_value.name] = self._y_value

        #set the given column to be the new Y value
        self._y_value = self._dataset[column]

        #update the x values to drop the new Y from X
        self._x_values = self._x_values.drop(columns=[column])

    @requireDataset
    def addConstant(self):
        """
        Creates a vector of ones and adds it to the X matrix.
        """

        #find the amount of rows the X matrix has, so we know how long the vector should be.
        num_rows = self._x_values.shape[0]

        #adds a series of 1s to the constants vector, using the number of rows.
        constants_vector = np.repeat(1, num_rows)

        #include the new column in the X matrix
        self._x_values['constants'] = constants_vector

    @requireDataset
    def xTranspose(self):
        """
        Transposes the X matrix.
        Returns: the transposed version of the X matrix.
        """

        x_transposed = np.transpose(self._x_values)
        return x_transposed

    #abstract method to be implemented within each subclass
    def loadData(self):
        """Abstract method to load data. Must be implemented in each subclass with its spesific behavior."""
        raise NotImplementedError


class CSVLoader(DataLoader):
    """
    Constructor for the CSV subclass.

    Parameters:
        dataset_name: the name of the dataset including .csv format.
    """

    def __init__(self,dataset_name):
        self._dataset_name = dataset_name
        super().__init__()

    def loadData(self):
        """Loads the dataset using csv format from Pandas."""

        try:
            data = pd.read_csv(self._dataset_name)
            self._dataset = data

        except Exception:
            print("Could not load CSV dataset")
        

class SMLoader(DataLoader):
    """
    Constructor for the SM subclass.

    Parameters:
        dataset_name: name of the dataset.
        library: name of the library to load from.
    """

    def __init__(self,dataset_name,library):

        self._library = library
        self._dataset_name = dataset_name
        super().__init__()

    def loadData(self):
        """Loads datasets from stats models using their format."""

        try:
         data = sm.datasets.get_rdataset(self._dataset_name,self._library).data
         self._dataset = data

        except Exception:
         print("Could not load SM dataset")

        
class WebLoader(DataLoader):
    """
    Constructor for the Web subclass.

    Parameters:
        url: url string to the dataset.
    """

    def __init__(self,url):
        super().__init__()
        self._url = url

    def loadData(self):
        """Loads data using url format."""
        try:
         self._dataset = pd.read_csv(self._url)
        except Exception:
            print("Could not load dataset from URL")


