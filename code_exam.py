"""Simple regression code example"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt



def load_data(fname):
    df = pd.DataFrame.from_csv(fname,sep='\t')    
    
    return df
    
def summary(r):
     print "Min, Mean, Max" , np.min(r), np.mean(r), np.max(r)
    
     print "1stQu, 3stQu, Median", np.percentile(r, [25, 75]), np.median(r)
    
    
def make_regression(fname="./data/prostate.data.txt"):
    
    from pandas.tools.plotting import scatter_matrix
    
    data = load_data(fname)   
    
    scatter_matrix(data)
    
    print data.describe(percentiles = [.25, .5, .75],)       
    
    print "\n\n\n******************* Split data set  ******************************"
    
    train = data[data.train == 'T'].drop('train',1)
    
    test = data[data.train == 'F'].drop('train',1)          
    
    train['intercept'] = np.ones((len(train), ))

    test['intercept'] = np.ones((len(test), ))
    
    x_train = train.drop('lpsa',1)

    x_test = test.drop('lpsa',1)
    
    y_train = train['lpsa']

    y_test = test['lpsa']
    
    print "\n\n\n******************* Make regression *********************************"
    
    res = sm.OLS(y_train, x_train).fit()

    print res.summary()
    
    print "\n\n\n******************* Prediction Error Stat ******************************"
    
    predict = res.predict(x_test)

    err = (y_test - predict)**2
    
    summary(err)  
    
def make_knn(neighbors,fname="./data/prostate.data.txt"):
    
    from sklearn.neighbors import KNeighborsRegressor
    
    from sklearn import preprocessing
    
    import numpy as np

    
    data = load_data(fname)
            
    train = data[data.train == 'T'].drop('train',1)
    
    test = data[data.train == 'F'].drop('train',1)
    
    train_X = train.drop('lpsa',1)
       
    train_Y = train.ix[:,"lpsa"]
    
    test_X = test.drop('lpsa',1)
    
    test_Y = test.ix[:,"lpsa"]
    
    # variables should be standardized for properly applying k-NN!
    
    train_X = preprocessing.scale(train_X)
    
    test_X = preprocessing.scale(test_X)
    
    neigh = KNeighborsRegressor(n_neighbors=neighbors)
    
    fit = neigh.fit(train_X, train_Y)
    
    pred = fit.predict(test_X)
    
    err = (test_Y - pred) **2
    
    summary( err)
    
    return np.mean(err)
    
    
def run_knn_test():

    v  =[]
    
    for n in range(1,30,1):
        v.append(make_knn(n))
        
    plt.plot(v)
    

    
    
