import sqlite3
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
import statsmodels.api as sm 
from update import seqnum

np.set_printoptions(precision=4, suppress=True)

# predict pm2.5 value
def predict(y,m,d):
    # get data from database
    print "loading data from database:"
    conn = sqlite3.connect('airweather.db')
    c = conn.cursor()
    c.execute("""select PM25, temperature, humidity, airpressure,
              wind_speed, wind_direction, dew, rain from PM25""")
    res = c.fetchall()
    num = 0
    numtotal = seqnum(y,m,d,23,30)
    dataset = []
    for row in res:
        if num%2 ==0:
            dataset.append(list(row))
        num += 1
        if num == numtotal:
            break
    conn.close()
    dataset = np.array(dataset,dtype = np.str)
    print dataset
    print "Done!"

    print "preprocessing the input data:"
    # complete the ommited observations
    idx = np.where(dataset == 'None')
    for i in np.unique(idx[0]):
        dataset[i,1:] = dataset[i-1,1:]
    # interpolate PM25 column in dataset
    idx = np.where(dataset == '')
    dataset[idx] = np.nan
    pm25 = np.double(dataset[:,0])
    ok = -np.isnan(pm25)
    xp = ok.nonzero()[0]
    fp = pm25[ok]
    x = np.isnan(pm25).nonzero()[0]
    pm25[np.isnan(pm25)] = np.interp(x,xp,fp)
    dataset = np.hstack((pm25[:,np.newaxis],dataset[:,1:]))
    # alter wind_direction into discrete numbers
    wind_dir = dataset[:,5]
    wind_dir[wind_dir=='None'] = 'Variable'
    wind_dir[np.array(wind_dir=='ENE') + np.array(wind_dir=='NNE')] = 'NE'
    wind_dir[np.array(wind_dir=='ESE') + np.array(wind_dir=='SSE')] = 'SE'
    wind_dir[np.array(wind_dir=='NNW') + np.array(wind_dir=='WNW')] = 'NW'
    wind_dir[np.array(wind_dir=='SSW') + np.array(wind_dir=='WSW')] = 'SW'
    N = np.array(wind_dir=='North', dtype = np.double)
    S = np.array(wind_dir=='South', dtype = np.double)
    E = np.array(wind_dir=='East', dtype = np.double)
    W = np.array(wind_dir=='West', dtype = np.double)
    NE = np.array(wind_dir=='NE', dtype = np.double)
    SE = np.array(wind_dir=='SE', dtype = np.double)
    NW = np.array(wind_dir=='NW', dtype = np.double)
    SW = np.array(wind_dir=='SW', dtype = np.double)
    V = np.array(wind_dir=='Variable', dtype = np.double)
    dataset = np.hstack((dataset[:,(0,1,2,3,4,6,7)],N[:,np.newaxis],S[:,np.newaxis],E[:,np.newaxis],W[:,np.newaxis],
                   NE[:,np.newaxis],SE[:,np.newaxis],NW[:,np.newaxis],SW[:,np.newaxis],V[:,np.newaxis]))
    dataset = np.double(dataset)
    print dataset
    # generate the dataset for regression X,y
    delay = 24
    length = len(dataset)
    y = np.mean(dataset[delay:,0].reshape((length-delay)/24,24),axis=1) #24 hour average||daily average
    X = dataset[delay::24,1:]
    for i in range(1,delay):
        X = np.hstack((X,dataset[delay-i:-i:24])) # previous 48 hours observations
    print "Done!"

    # train model according to X,y steming from database data
    print "Training the model:"

    # sklearn.linear_model.lasso
    lasso = linear_model.LassoCV(alphas=None, fit_intercept = True, cv=10, verbose=True)
    lasso.fit(X,y)
    print lasso.score(X,y)

    # linear regression
##    pca = PCA(n_components = 0.95)
##    pca.fit(X)
##    print X.shape
##    X = pca.fit_transform(X)
##    print X.shape
##    X = sm.add_constant(X, prepend=False)
##    ols = sm.OLS(y,X)
##    results = ols.fit()
##    print results.summary()

    # ANN
    print "Done!"
    
    arg=X[-8:]    
    return lasso.predict(arg)
