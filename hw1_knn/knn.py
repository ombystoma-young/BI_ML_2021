import numpy as np
from scipy import stats


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=1):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        
        shape = (np.shape(X)[0], np.shape(self.train_X)[0])
        dist_mat = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                dist_mat[i, j] += np.sum(abs(X[i] - self.train_X[j]))
        return dist_mat
        


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        shape = (np.shape(X)[0], np.shape(self.train_X)[0])
        dist_mat = np.zeros(shape)

        for i in range(shape[0]):
            dist_mat[i] += np.sum(abs(X[i] - self.train_X), axis=1)
        
        return dist_mat


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
           
        It also works if X_train shape < X_test shape. 

        """

        
        X_shape = np.shape(X)
        X_reshaped = X.reshape(X_shape[0], 1, X_shape[1])
        dist_mat = np.sum(abs(X_reshaped - self.train_X), axis=2)
        
        return dist_mat
        

    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
      

        distances_T = distances.T
        nearest_neighbor_ids = distances_T.argsort(axis=0)[:self.k]

#        Изначально я пыталась реализовать следующий код, он работает в пайчарме, а в джупитере - нет. :(        
#        values, counts = np.unique(self.train_y[nearest_neighbor_ids], axis=0, return_counts=True)
#        prediction = values[np.argmax(counts)]


        pred = stats.mode(self.train_y[nearest_neighbor_ids], axis=0)[0]  
        prediction = pred.reshape(n_test,)
       
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]

        distances_T = distances.T
        nearest_neighbor_ids = distances_T.argsort(axis=0)[:self.k]
        
        pred = stats.mode(self.train_y[nearest_neighbor_ids], axis=0)[0]  
        prediction = pred.reshape(n_test,)
        
        return prediction
    