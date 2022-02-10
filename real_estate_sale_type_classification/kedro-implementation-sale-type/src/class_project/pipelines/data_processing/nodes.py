import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn import under_sampling, over_sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def add_features(raw_df):
    raw_df['TotalSF'] = raw_df['1stFlrSF']+raw_df['2ndFlrSF']+raw_df['TotalBsmtSF']
    raw_df['OverallQCP'] = (raw_df['OverallQual']*raw_df['OverallCond'])/100
    raw_df['TotalBath'] = raw_df['FullBath']+raw_df['BsmtFullBath']

    #create two quant features for location (relevant to price)...median is better
    hood_names = raw_df['Neighborhood'].unique()
    hood_avg_prices = []
    hood_median_prices = []
    for name in hood_names:
        df_temp = raw_df[(raw_df['Neighborhood']==name)]
        avg_price = int(df_temp['SalePrice'].mean())
        hood_avg_prices.append(avg_price)
        median_price = df_temp['SalePrice'].median()
        hood_median_prices.append(median_price)
    #     print("{} has an average home sale price of {}".format(name,avg_price))

    name_avg_price_dict = dict(zip(hood_names, hood_avg_prices))
    raw_df['HoodAvg'] = raw_df['Neighborhood'].map(lambda x: name_avg_price_dict[x])
    name_med_price_dict = dict(zip(hood_names, hood_median_prices))
    raw_df['HoodMed'] = raw_df['Neighborhood'].map(lambda x: name_med_price_dict[x])

    #create quant feature for Functional
    func_dict ={
        'Typ':7,
        'Min1':6,
        'Min2':5,
        'Mod':4,
        'Maj1':3,
        'Maj2':2,
        'Sev':1,
        'Sal':0  
    }
    raw_df['FuncScore'] = raw_df['Functional'].map(lambda x: func_dict[x])
    return raw_df

def standardize_foi(cols, foi_df):
    scaler = preprocessing.MinMaxScaler()
    for feature in cols:
        feature_mat = foi_df[feature].values.reshape(-1,1)
        foi_df.loc[:, feature] = scaler.fit_transform(feature_mat)
    return foi_df

def prepare_y(raw_df):
    raw_df.loc[raw_df['SaleCondition'] == 'Normal', 'SaleCondition'] = 0
    raw_df.loc[raw_df['SaleCondition'] != 0, 'SaleCondition'] = 1
    return raw_df['SaleCondition']

def create_X_y(raw_df, cols):
    all_f_df = add_features(raw_df)
    foi_df = all_f_df[list(cols)]
    s_foi_df = standardize_foi(cols, foi_df)

    X = s_foi_df.copy()
    y = prepare_y(raw_df)

    lab_enc = preprocessing.LabelEncoder()
    y_encoded = lab_enc.fit_transform(y)
    return X, y_encoded

def log_reg_models(X, y_encoded):
    #imbalanced
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)
    y_pred = model_lr.predict(X_test)
    lr_imb_acc = accuracy_score(y_test, y_pred)
    lr_imb_f1 = f1_score(y_test, y_pred)

    #undersampling
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    rus = RandomUnderSampler(random_state=0)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    model_lr_u = LogisticRegression()
    model_lr_u.fit(X_train_rus, y_train_rus)
    y_pred = model_lr_u.predict(X_test)
    lr_under_acc = accuracy_score(y_test, y_pred)
    lr_under_f1 = f1_score(y_test, y_pred)

    #undersampling, weighted 0.82 (minority class to majority class)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    rus_weighted = RandomUnderSampler(random_state=0, sampling_strategy=0.82)
    X_train_rus, y_train_rus = rus_weighted.fit_resample(X_train, y_train)
    model_lr_u = LogisticRegression()
    model_lr_u.fit(X_train_rus, y_train_rus)
    y_pred = model_lr_u.predict(X_test)
    lr_under_w_acc = accuracy_score(y_test, y_pred)
    lr_under_w_f1 = f1_score(y_test, y_pred)

    #oversampling
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    ros = RandomOverSampler(random_state=0)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    model_lr_o = LogisticRegression()
    model_lr_o.fit(X_train_ros, y_train_ros)
    y_pred = model_lr_o.predict(X_test)
    lr_over_acc = accuracy_score(y_test, y_pred)
    lr_over_f1 = f1_score(y_test, y_pred)

    #oversampling, weighted 0.97 (minority class to majority class)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    ros_weighted = RandomOverSampler(random_state=0, sampling_strategy=.97)
    X_train_ros, y_train_ros = ros_weighted.fit_resample(X_train, y_train)
    model_lr_o = LogisticRegression()
    model_lr_o.fit(X_train_ros, y_train_ros)
    y_pred = model_lr_o.predict(X_test)
    lr_over_w_acc = accuracy_score(y_test, y_pred)
    lr_over_w_f1 = f1_score(y_test, y_pred)

    lr_model_scores = pd.DataFrame(
        {
            "Model": ["log reg imbalanced", "log reg undersampled", "log reg undersampled weighted", "log reg oversampled", "log reg oversampled weighted"],
            "Accuracy": [lr_imb_acc, lr_under_acc, lr_under_w_acc, lr_over_acc, lr_over_w_acc],
            "F1": [lr_imb_f1, lr_under_f1, lr_under_w_f1, lr_over_f1, lr_over_w_f1],
        },
        index=None,
    )
    return lr_model_scores

def rand_for_models(X, y_encoded):
    #imbalanced
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    rfc = RandomForestClassifier(n_estimators=13, random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    rf_imb_acc = accuracy_score(y_test, y_pred)
    rf_imb_f1 = f1_score(y_test, y_pred)

    #undersampling
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    rus = RandomUnderSampler(random_state=0)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    rfc_u = RandomForestClassifier(n_estimators=13, random_state=0)
    rfc_u.fit(X_train_rus, y_train_rus)
    y_pred = rfc_u.predict(X_test)
    rf_under_acc = accuracy_score(y_test, y_pred)
    rf_under_f1 = f1_score(y_test, y_pred)

    #undersampling, weighted 0.52 (minority class to majority class)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    rus = RandomUnderSampler(random_state=0, sampling_strategy=0.52)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    rfc_u = RandomForestClassifier(n_estimators=21, random_state=0)
    rfc_u.fit(X_train_rus, y_train_rus)
    y_pred = rfc_u.predict(X_test)
    rf_under_w_acc = accuracy_score(y_test, y_pred)
    rf_under_w_f1 = f1_score(y_test, y_pred)

    #oversampling
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    ros = RandomOverSampler(random_state=0)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    rfc_o = RandomForestClassifier(n_estimators=13, random_state=0)
    rfc_o.fit(X_train_ros, y_train_ros)
    y_pred = rfc_o.predict(X_test)
    rf_over_acc = accuracy_score(y_test, y_pred)
    rf_over_f1 = f1_score(y_test, y_pred)

    #oversampling, weighted 0.97 (minority class to majority class)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    ros_weighted = RandomOverSampler(random_state=0, sampling_strategy=0.97)
    X_train_ros, y_train_ros = ros_weighted.fit_resample(X_train, y_train)
    rfc_o = RandomForestClassifier(n_estimators=7, random_state=0)
    rfc_o.fit(X_train_ros, y_train_ros)
    y_pred = rfc_o.predict(X_test)
    rf_over_w_acc = accuracy_score(y_test, y_pred)
    rf_over_w_f1 = f1_score(y_test, y_pred)

    rf_model_scores = pd.DataFrame(
        {
            "Model": ["random forest imbalanced", "random forest undersampled", "random forest undersampled weighted", "random forest oversampled", "random forest oversampled weighted"],
            "Accuracy": [rf_imb_acc, rf_under_acc, rf_under_w_acc, rf_over_acc, rf_over_w_acc],
            "F1": [rf_imb_f1, rf_under_f1, rf_under_w_f1, rf_over_f1, rf_over_w_f1],
        },
        index=None,
    )
    return rf_model_scores

def neu_net_models(X, y_encoded):
    #imbalanced
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    nn = MLPClassifier(max_iter=1000, activation="relu", hidden_layer_sizes=(10,10))
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    nn_imb_acc = accuracy_score(y_test, y_pred)
    nn_imb_f1 = f1_score(y_test, y_pred)

    #undersampling
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    rus = RandomUnderSampler(random_state=0)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    nn = MLPClassifier(max_iter=1000, activation="relu", alpha=0.01, hidden_layer_sizes=(10,10))
    nn.fit(X_train_rus, y_train_rus)
    y_pred = nn.predict(X_test)
    nn_under_acc = accuracy_score(y_test, y_pred)
    nn_under_f1 = f1_score(y_test, y_pred)

    #undersampling, weighted 0.59 (minority class to majority class)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    rus_weighted = RandomUnderSampler(random_state=0, sampling_strategy=0.57)
    X_train_rus, y_train_rus = rus_weighted.fit_resample(X_train, y_train)
    nn = MLPClassifier(max_iter=1100, activation="relu", alpha=0.01, hidden_layer_sizes=(19,19))
    nn.fit(X_train_rus, y_train_rus)
    y_pred = nn.predict(X_test)
    nn_under_w_acc = accuracy_score(y_test, y_pred)
    nn_under_w_f1 = f1_score(y_test, y_pred)

    #oversampling
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    ros = RandomOverSampler(random_state=0)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    nn = MLPClassifier(max_iter=1000, activation="relu", alpha=0.01, hidden_layer_sizes=(10,10))
    nn.fit(X_train_ros, y_train_ros)
    y_pred = nn.predict(X_test)
    nn_over_acc = accuracy_score(y_test, y_pred)
    nn_over_f1 = f1_score(y_test, y_pred)

    #oversampling, weighted
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    ros_weighted = RandomOverSampler(random_state=0, sampling_strategy=0.72)
    X_train_ros, y_train_ros = ros_weighted.fit_resample(X_train, y_train)
    nn = MLPClassifier(max_iter=1000, activation="relu", alpha=0.01, hidden_layer_sizes=(17,17))
    nn.fit(X_train_ros, y_train_ros)
    y_pred = nn.predict(X_test)
    nn_over_w_acc = accuracy_score(y_test, y_pred)
    nn_over_w_f1 = f1_score(y_test, y_pred)

    nn_model_scores = pd.DataFrame(
        {
            "Model": ["neural network imbalanced", "neural network undersampled", "neural network undersampled weighted", "neural network oversampled", "neural network oversampled weighted"],
            "Accuracy": [nn_imb_acc, nn_under_acc, nn_under_w_acc, nn_over_acc, nn_over_w_acc],
            "F1": [nn_imb_f1, nn_under_f1, nn_under_w_f1, nn_over_f1, nn_over_w_f1],
        },
        index=None,
    )
    return nn_model_scores