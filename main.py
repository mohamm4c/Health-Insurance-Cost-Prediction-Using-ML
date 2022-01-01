import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score,cross_val_predict
import seaborn as sns
from math import sqrt
data1=pd.read_csv('insurance.csv')
data = pd.read_csv('insurance.csv')

number=LabelEncoder()
data['sex']=number.fit_transform(data['sex'].astype('str'))
data['smoker']=number.fit_transform(data['smoker'].astype('str'))
regions = pd.get_dummies(data['region'])
data[['northeast','northwest','southeast','southwest']]=regions[['northeast','northwest','southeast','southwest']]
data = data.drop('region',axis=1)
data.head()


data[['age','sex','bmi','children','smoker','charges','northeast','northwest','southeast','southwest']] = StandardScaler().fit_transform(data[['age','sex','bmi','children','smoker','charges','northeast','northwest','southeast','southwest']])


Y = data.charges

X = data.drop('charges',axis=1)
#print(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=0)
SupportVector = svm.SVR(kernel='rbf')
SupportVector.fit(X_train, Y_train)
SVM_pred=SupportVector.predict(X_test)
#Score=SupportVector.score(X_test, Y_test)
r2scoreSVM=r2_score(Y_test, SVM_pred)
CSVM = cross_val_score(SupportVector, X, Y, cv=10)
MSESVM=mean_squared_error(Y_test, SVM_pred)
RMSESVM=sqrt(MSESVM)
meanAESVM=mean_absolute_error(Y_test, SVM_pred)
print(CSVM.mean(),CSVM.std())


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, Y_train)
linearpred=linreg.predict(X_test)
r2scorelinear=r2_score(Y_test,linearpred)
Clinear = cross_val_score(linreg, X, Y, cv=10)
MSELinear=mean_squared_error(Y_test, linearpred)
meanAELinear=mean_absolute_error(Y_test, linearpred)
RMSELinear=sqrt(MSELinear)
print(Clinear.mean(),Clinear.std())
#accuracy = accuracy_score(Y_test, y_pred)

from sklearn import tree
tre=tree.DecisionTreeRegressor(criterion='mse',min_samples_split=2,
 min_samples_leaf=2,
 max_features='sqrt',
 max_depth=8,
 random_state=42)
tre.fit(X_train, Y_train)
trepred=tre.predict(X_test)
r2scoreDT=r2_score(Y_test, trepred)
CDT = cross_val_score(tre, X, Y, cv=10)
MSEDT=mean_squared_error(Y_test, trepred)
meanAEDT=mean_absolute_error(Y_test, trepred)
RMSEDT=sqrt(MSEDT)
print(CDT.mean(),CDT.std())

from sklearn.ensemble import RandomForestRegressor
RFreg=RandomForestRegressor(n_estimators= 700,
 min_samples_split=2,
 min_samples_leaf=2,
 max_features='log2',
 max_depth=11,
 bootstrap=True)
RFreg.fit(X_train,Y_train)
RFpred=RFreg.predict(X_test)
r2scoreRF=r2_score(Y_test,RFpred)
MSERF=mean_squared_error(Y_test, RFpred)
meanAERF=mean_absolute_error(Y_test, RFpred)
RMSERF=sqrt(MSERF)
CRF = cross_val_score(RFreg, X, Y, cv=10)
print(CRF.mean(),CRF.std())

from sklearn.ensemble import GradientBoostingRegressor
GRreg=GradientBoostingRegressor(loss='huber')
GRreg.fit(X_train,Y_train)
GRpred=GRreg.predict(X_test)
r2scoreGR=r2_score(Y_test,GRpred)
MSEGR=mean_squared_error(Y_test, GRpred)
RMSEGR=sqrt(MSEGR)
meanAEGR=mean_absolute_error(Y_test, GRpred)

CGR = cross_val_score(GRreg, X, Y, cv=10)
print(CGR.mean(),CGR.std())



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
pca_test = PCA(n_components=9)
pca_test.fit(X_train)
sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axvline(linewidth=4, color='r', linestyle = '--', x=7, ymin=0, ymax=1)
plt.show()
evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(pca_test.explained_variance_ratio_)
pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr
pca_df.head(10)

pca = PCA(n_components=8)
pca.fit(X_train)
X_train_scaled_pca = pca.transform(X_train)
X_test_scaled_pca = pca.transform(X_test)

RFregPCA=RandomForestRegressor(n_estimators= 700,#with PCA and HYperTuning
 min_samples_split=2,
 min_samples_leaf=2,
 max_features='log2',
 max_depth=11,
 bootstrap=True)
RFregPCA.fit(X_train_scaled_pca,Y_train)
RFpredPCA=RFregPCA.predict(X_test_scaled_pca)
r2scoreRFPCA=r2_score(Y_test,RFpredPCA)
CRFPCA = cross_val_score(RFregPCA, X, Y, cv=10)
MSERFPCA=mean_squared_error(Y_test, RFpredPCA)
meanAERFPCA=mean_absolute_error(Y_test, RFpredPCA)
RMSERFPCA=sqrt(MSERFPCA)
print(CRFPCA.mean(),CRFPCA.std())

g=plt.scatter(Y_test,GRpred)
g.axes.set_xlabel('True Values ')
g.axes.set_ylabel('Predictions ')

plt.scatter(SVM_pred,Y_test,color='red')
#sns.set_style("whitegrid");
#sns.pairplot(data1);
#plt.show()

data1.hist(bins=30,figsize=(20,20))
fig=plt.figure(figsize=(12,6))
sns.heatmap(data1.corr(),annot=True)
sns.distplot(data1['charges'])



'''n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rs = RandomizedSearchCV(rfc_2, 
                        param_dist, 
                        n_iter = 100, 
                        cv = 3, 
                        verbose = 1, 
                        n_jobs=-1, 
                        random_state=0)
rs.fit(X_train_scaled_pca, y_train)
rs.best_params_'''



