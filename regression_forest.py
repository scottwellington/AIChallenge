import os
import numpy as np
import cv2
import random
import pandas as pd
import pdb
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectPercentile
from sklearn.decomposition import TruncatedSVD
from sklearn import tree
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error


class LinearRegression:

    def __init__(self):

        # store current least squares error and step size:
        self.error = np.inf # initialise as inf
        self.step_size = 0.1

        # set hyperperamters:
        self.num_iterations = 10000
        self.converge_threshold = 1e-8

        # where we are learning the function y = wx + c:
        self.w = 1.5
        self.c = 0.5

        # convergance flag (set to True when converged)
        self.converged = False


    def least_squares_error(self, x, y, w, c):

        # OLD, SLOWER METHOD:
        #squared_error = np.sum([(y[i] - w*x[i] - c)**2 for i in range(len(x))])
    
        # NEW, FASTER METHOD:
        squared_error = np.sum((np.transpose([y]) - w*x - c)**2)

        return squared_error


    def calc_gradients_for_least_squares(self, x, y, w, c):

    # calculate gradients wrt the parameters

        # NEW, FASTER METHOD:
        grad_w = np.sum(2*(x*((w*x) + c - np.transpose([y]))))
        grad_c = np.sum(2*((w*x) + c - np.transpose([y])))

        # OLD, SLOWER METHOD:
        # grad_w = np.sum([(2*(x[i]*((w*x[i]) + c - y[i]))) for i in range(len(x))])
        # grad_c = np.sum([(2*((w*x[i]) + c - y[i])) for i in range(len(x))])
        
        return grad_w, grad_c


    def run_iteration(self, x, y):
        
        try:
            # Initilise least squares error:
            self.error = self.least_squares_error(x, y, self.w, self.c)
        except Exception as err:
            exit(f'Error defining training data: {err}. Exiting...')

        # Evaluate the gradients:
        grad_w, grad_c = self.calc_gradients_for_least_squares(x, y, self.w, self.c)

        # Take a step in the direction of the negative gradient proportional to the step size:
        w_new, c_new = self.w - (self.step_size * grad_w), self.c - (self.step_size * grad_c)

        # Evaluate and remember the squared error..
        error_new = self.least_squares_error(x, y, w_new, c_new)

        # Check that error is decreasing and reduce step size if not:
        if error_new > self.error:
            self.step_size = self.step_size - self.step_size * 0.9
            self.run_iteration(x, y)
            
        # check for convergence and terminate the loop if converged:

        if abs(self.error - error_new) <= self.converge_threshold:
            self.converged = True
                
        # Take the step...
        self.w = w_new
        self.c = c_new
        self.error = error_new

    def main(self, x_train, y_train, x_val, y_val, x_test, y_test):

        train_RMSE, test_RMSE = 0, 0

        for iteration in range(self.num_iterations):

            self.run_iteration(x_train, y_train)
            
            print('iteration %4d, Error = %f, w = %f, c = %f' % (
                iteration, self.error, self.w, self.c))
            
            if self.converged:
                # Break out of iteration loop..
                print('Converged!')
                break

        test_error = self.least_squares_error(x_test, y_test, self.w, self.c)           

        train_RMSE = np.sqrt(self.error/len(y_train))
        test_RMSE = np.sqrt(test_error/len(y_test))

        print('\nAfter gradient descent optimisation:')
        print(f'Optimised w = {self.w}')
        print(f'Optimised c = {self.c}')
        # print(f'Squared error (linear least squares) on training set = {self.error}')
        # print(f'Squared error (linear least squares) on test set = {test_error}')
        print(f'RMSE on training set = {train_RMSE}')
        print(f'RMSE on test set = {test_RMSE}')

        return train_RMSE, test_RMSE


class RegressionForest:

    def __init__(self, ensemble_size=10, max_depth=10):

        # decide size of ensemble (also determines the number of bootstrap draws)
        
        self.ensemble = True
        self.random_subspace = True

        # set hyperperameters:
        
        self.max_depth = max_depth
        self.ensemble_size = ensemble_size

    def find_split(self, x, y):
        # Given a dataset and its target values, find the optimal combination
        # of feature and split points that gives the minimum variance.
        
        # Starting variance (to measure improvement):
        start_variance = np.var(y)
        
        # Initialise:
        best = {'weighted_variance': start_variance}
        idx = range(x.shape[1])

        if self.random_subspace:

            # set random subspace bootstrap size to be √(no. of features) (rounded up):
            random_subspace_bootstrap_size = int(np.ceil(np.sqrt(x.shape[1])))
            # make that many bootstrap draws of the features (without replacement):
            random_subspace_bootstrap_draws = np.random.choice(range(x.shape[1]),
                size=random_subspace_bootstrap_size, replace=False)
            # replace our indices with the bootstrap draws:
            idx = sorted(random_subspace_bootstrap_draws)

        # Loop every possible split of every (bootstrapped) dimension:

        for i in idx:
            for split in np.unique(x[:,i]):
                
                left_indices = [v for v in range(len(x[:,i])) if x[v,i] <= split]
                right_indices = [v for v in range(len(x[:,i])) if x[v,i] > split]
                
                left_exemplars = [y[v] for v in left_indices]
                right_exemplars = [y[v] for v in right_indices]
                
                left_variance = np.var(left_exemplars)
                right_variance = np.var(right_exemplars)
                
                n = len(x[:,i])
                l = (len(left_exemplars)/n)*left_variance
                r = (len(right_exemplars)/n)*right_variance
                
                weighted_variance = l + r # want to minimise this
                
                if weighted_variance < best['weighted_variance']:
                    best = {'feature' : i,
                            'split' : split,
                            'weighted_variance' : weighted_variance, 
                            'left_indices' : left_indices,
                            'right_indices' : right_indices}
        return best

    def build_tree(self, x, y, max_depth = np.inf):
        
        # Check if either of the stopping conditions have been reached. If so generate a leaf node:
        
        if max_depth==1 or (y==y[0]).all():
        
            # Generate a leaf node.
            # for Regression, take the mean of the values that reached this leaf,
            # not the dominant class label (as per classification)
            return {'leaf': True, 'value': np.mean(y)}
        
        else:
            move = self.find_split(x, y)
            
            max_depth-=1 # iterative decrease, per depth

            left = self.build_tree(x[move['left_indices'],:], y[move['left_indices']], max_depth - 1)
            right = self.build_tree(x[move['right_indices'],:], y[move['right_indices']], max_depth - 1)
            
            return {'leaf' : False,
                    'feature' : move['feature'],
                    'split' : move['split'],
                    'weighted_variance' : move['weighted_variance'],
                    'left' : left,
                    'right' : right}

    def cart(self, tree, samples):

        # predict value for every entry of a data matrix:

        ret = np.empty(samples.shape[0], dtype=np.float64)
        ret.fill(-1)
        indices = np.arange(samples.shape[0])
        
        def tranverse(node, indices):
            nonlocal samples
            nonlocal ret
            
            if node['leaf']:
                ret[indices] = node['value']
            
            else:
                going_left = samples[indices, node['feature']] <= node['split']
                left_indices = indices[going_left]
                right_indices = indices[np.logical_not(going_left)]
                
                if left_indices.shape[0] > 0:
                    tranverse(node['left'], left_indices)
                    
                if right_indices.shape[0] > 0:
                    tranverse(node['right'], right_indices)
        
        tranverse(tree, indices)
        return ret

    def fit(self, x, y):

        self.tree_dict = defaultdict()
        time = datetime.datetime.now().timestamp() * 1000

        for i in range(self.ensemble_size):

            # Select S, the size of the emsemble and the number of bootstrap draws:
            bootstrap_aggregate_draw_size = self.ensemble_size

            bootstrap_aggregate_draws = sorted(np.random.choice(range(x.shape[0]),
                size=bootstrap_aggregate_draw_size, replace=False))

            x = np.array([x[i,:] for i in bootstrap_aggregate_draws])
            y = np.array([y[i] for i in bootstrap_aggregate_draws])

            tree = self.build_tree(x, y, max_depth=self.max_depth)
            tree_dict[i] = tree

        # Store times for reporting:
        time = (datetime.datetime.now().timestamp() * 1000) - time

        return time

    def predict(self, x):

        predicted = []

        time = datetime.datetime.now().timestamp() * 1000

        # Get each tree in our ensemble to output a prediction of the dataset passed to it:
        for i in self.tree_dict.keys():

            time_a = datetime.datetime.now().timestamp() * 1000
            predicted.append(self.cart(tree_dict[i], x))
            time_b = datetime.datetime.now().timestamp() * 1000
            time += time_b - time_a

        # Take the transpose of this output, so that each row of the matrix is every tree's
        # guess for that row's target value (e.g. row1 = every tree's guess for row1,
        # row2 = every tree's guess for row2, etc...)
        predicted = np.array(predicted).T

        # take the mean value of the guesses:
        predicted = [np.mean(i) for i in predicted]

        return time, predicted


class BoostingRegressor():
    
    # Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). AdaBoost. Statistics and its Interface, 2(3), 349-360.

    def __init__(self,
                 max_depth = 2,
                 weak_learners = 10000,
                 bootstrap = True,
                 random_subspace = False):

        self.models = [] # we'll store our model.fits here
        
        self.weak_learners = weak_learners
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.random_subspace = random_subspace

     
    def fit(self, x_train, y_train):
        
        # initialise the weight matrices:
        sample_weights = np.full(x_train.shape[0], fill_value = 1/x_train.shape[0], dtype=float) # Sample weights
        self.model_weights = np.zeros(self.weak_learners, dtype=float) # Weak learner weights
        
        # convenience handlers:
        self.total_classes = len(np.unique(y_train))
        self.class2idx = {j:i for i,j in enumerate(np.unique(y_train))}
        self.idx2class = {i:j for i,j in enumerate(np.unique(y_train))}
        
        print('Training weak learners...')
        for i in tqdm(range(self.weak_learners)):
            
            x, y  = x_train, y_train
            
            if self.bootstrap:
            
                bootstrap_aggregate_draws = sorted(np.random.choice(range(x.shape[0]),
                    size=int(np.ceil(np.sqrt(x.shape[0]))), replace=False))

                x = np.array([x[i,:] for i in bootstrap_aggregate_draws])
                y = np.array([y_train[i] for i in bootstrap_aggregate_draws])

            if self.random_subspace:
                x = self.get_random_subspace(x)
            
            RF = RegressionForest(max_depth=self.max_depth, ensemble_size=self.weak_learners)
            self.models.append(RF.fit(x,y))
        
        if self.random_subspace:
            x_train = self.get_random_subspace(x_train)        
        
        print('Predicting SGR...')
        for i in tqdm(range(self.weak_learners)):
            
            model = self.models[i]
            _, pred = model.predict(x_train) # get model predictions               

            e = np.abs(y_train-pred)

            self.models[i] = (i,model,pred,e) # store this model with its predictions and absolute errors
            
        self.models.sort(key=lambda tup: np.sum(tup[3])) # sorts weak learners in place, from strongest to weakest

        print('Boosting weak learners...')
        for i in tqdm(range(self.weak_learners)):
            
            model_index = self.models[i][0] # weak learner index
            model = self.models[i][1] # weak learner
            model_p = self.models[i][2] # weak learner predictions
            model_e = self.models[i][3] # weak learner absolute errors
            
            weighted_e = np.sum(model_e * sample_weights)/np.sum(sample_weights)
            err = np.log(1/(weighted_e + 1e-10) - 1) # add some noise to prevent devision by zero errors
            
            self.model_weights[model_index] = err if err > 0 else 0 # if error is above 0, set this learner's contributions
        
            alpha = np.full(x_train.shape[0], fill_value = self.model_weights[model_index], dtype=float)
            
            sample_weights *= np.exp(alpha * model_e)
            sample_weights /= np.sum(sample_weights) # normalise sample weights
        
        self.model_weights /= np.sum(self.model_weights) # normalise weak learner weights
        
    def get_random_subspace(self, x):
        
        # set random subspace bootstrap size to be √(no. of features) (rounded up):
        draws = int(np.ceil(np.sqrt(x.shape[1])))
        _x = []

        for j in range(x.shape[0]):

            random_subspace = sorted(np.random.choice(range(x.shape[1]), size=draws, replace=False))
            _x.append(x[j][random_subspace])
        
        return np.array(_x)
        
    def predict(self, x):        
        
        if self.random_subspace:
            x = self.get_random_subspace(x)
        
        preds = []
        
        for i in range(self.weak_learners):
            
            model_index = self.models[i][0] # weak learner index
            model = self.models[i][1] # weak learner
            model_p = self.models[i][2] # weak learner predictions
            model_e = self.models[i][3] # weak learner absolute errors

            pred = model.predict(x)
            pred *= self.model_weights[model_index] # weight the weak learner's predictions
            preds.append(pred)
        
        preds = np.array(preds)

        p_store = []

        for i in range(preds.shape[1]):
            p_store.append(np.mean(preds[:,i]))

        return np.array(p_store)


def preprocess_data(X_train, y_train, X_test, y_test):
    
    # Fit PCA to training data; apply fitted transform to test data:

    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # Shuffle the data with forced interleaving:
    
    labs = {lab: [] for lab in np.unique(y_train)}
    
    for i in range(y_train.shape[0]):
        labs[y_train[i]].append(i)
        
    for i in labs:
        random.shuffle(labs[i])
    
    idxs = [] 
    
    while labs:
        try:
            for i in labs:
                try:
                    idxs.append(labs[i].pop(0))
                except:
                    del labs[i]
        except:
            pass
    
    X_train = np.array(X_train[idxs])
    y_train = np.array(y_train[idxs])
            
    return (X_train, y_train, X_test, y_test) 


if __name__ == "__main__":

        df = pd.read_csv(os.path.join('.', 'df_all.csv'))

        y = df['r'] # target variable (SGR)
        X = df.drop(columns='r') # all predictor variables
        X = X.drop(columns='source') # all predictor variables
        X = X.drop(columns='Date') # all predictor variables

        y = np.array(y)
        X = np.array(X)

        kf = KFold(n_splits=5)
        folds = kf.get_n_splits(X)
        fold = 1

        lr_train_RMSE = []
        lr_test_RMSE = []

        train_preds = []
        test_preds = []

        pred_store = []
        error_store = []

        for train_index, test_index in kf.split(y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            (X_train, y_train, X_test, y_test) = preprocess_data(
                X_train, y_train, X_test, y_test)

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42)

            lr = LinearRegression()
            train_RMSE, test_RMSE = lr.main(X_train, y_train, X_val, y_val, X_test, y_test)

            lr_train_RMSE.append(train_RMSE)
            lr_test_RMSE.append(test_RMSE)

            br = BoostingRegressor()
            br.fit(X_train, y_train)

            # Take a test set-sized sample of the training set to get its accuracy:
            idxs = np.random.choice(
                range(X_train.shape[0]),size = X_test.shape[0], replace = False)
            _x, _y = np.array(
                [X_train[j,:] for j in idxs]), np.array([y_train[j] for j in idxs])

            x_pred = br.predict(_x)

            pred_store.append(x_pred)
            error_store.append(np.abs(_y - x_pred))

            acc = mean_squared_error(_y, x_pred, squared=False)
            print(f'fold {fold} RMSE:', acc)
            train_preds.append(acc)

            y_pred = br.predict(X_test)
            acc = mean_squared_error(y_test, y_pred, squared=False)
            print(f'fold {fold} RMSE:', acc)
            test_preds.append(acc)

            fold+=1

        print()
        print('Linear regression train RNSE', np.mean(lr_train_RMSE))
        print('Linear regression test RNSE', np.mean(lr_test_RMSE))    
        print()
        print('Boosted Regression Forest train RMSE', np.mean(train_preds))
        print('Boosted Regression Forest test RMSE', np.mean(test_preds))