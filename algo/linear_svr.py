import sys
import time

sys.path.insert(0, '../')
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVR
from rmsle import rmsle_cv

train_data = pd.read_csv("../data/train.csv", index_col=False)
test_data = pd.read_csv("../data/test.csv", index_col=False)

tr_id = train_data["Id"]
te_id = test_data["Id"]
tr_y = np.log1p(train_data["SalePrice"])
# te_y = test_data["SalePrice"]

train_data.drop("Id", axis=1, inplace=True)
test_data.drop("Id", axis=1, inplace=True)

distinct_values = {}
for column in train_data.columns:
    distinct_values[column] = train_data[column].nunique()
print(distinct_values)

ntrain = train_data.shape[0]
ntest = test_data.shape[0]
data = pd.concat((train_data, test_data)).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)

data_na = (data.isnull().sum() / len(data)) * 100
data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio': data_na})
missing_data.plot(kind="barh", figsize=(6, 8))

data["PoolQC"] = data["PoolQC"].fillna("None")
data["MiscFeature"] = data["MiscFeature"].fillna("None")
data["Alley"] = data["Alley"].fillna("None")
data["Fence"] = data["Fence"].fillna("None")
data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    data[col] = data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    data[col] = data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    data[col] = data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data[col] = data[col].fillna('None')
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data = data.drop(['Utilities'], axis=1)
data["Functional"] = data["Functional"].fillna("Typ")
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['MSSubClass'] = data['MSSubClass'].fillna("None")
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
# Check remaining missing values if any
data_na = (data.isnull().sum() / len(data)) * 100
data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)

# MSSubClass=The building class
data['MSSubClass'] = data['MSSubClass'].apply(str)

# Changing OverallCond into a categorical variable
data['OverallCond'] = data['OverallCond'].astype(str)

# Year and month sold are transformed into categorical features.
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(data[c].values))
    data[c] = lbl.transform(list(data[c].values))
print('Shape all_data: {}'.format(data.shape))

numeric_feats = data.dtypes[data.dtypes != "object"].index

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    data[feat] = boxcox1p(data[feat], lam)

dummy_data = pd.get_dummies(data)
tr_df = dummy_data[:ntrain]
te_df = dummy_data[ntrain:]

tr_X = tr_df.values
te_X = te_df.values

clf = GridSearchCV(LinearSVR(), {
    "C": [1e3, 1e4, 1e5, 1e2, 1],
    "epsilon": [0.1, 0.2, 0.3, 0.4, 0.01, 0.05, 0.001]
}, verbose=2, n_jobs=-1)
clf = clf.fit(tr_X, tr_y)
te_y = np.expm1(clf.predict(te_X))

res = pd.DataFrame({"Id": te_id, "SalePrice": te_y.reshape(te_y.shape[0])})
res.to_csv("../results/linear_svr_%s.csv" % time.time(), index=False)

score = rmsle_cv(clf, tr_X, tr_y)
print("Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
print(clf.cv_results_)
