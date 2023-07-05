from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def cross_validation(data):

    # Split data into equal partitions of size len_train
    
    num_train = 32 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40 # Length of each train-test set
    
    # Lists to store the results from each model
    rf_RESULTS = []
    xt_RESULTS = []
    
    i = 0
    
    # Models which will be used
    rf = RandomForestClassifier()
    xt = ExtraTreesClassifier()
    scaler = StandardScaler()
    
    while True:
        
        # Partition the data into chunks of size len_train every num_train days
        df_train = data.iloc[i * num_train : (i * num_train) + len_train]
        print(f"window: {i * num_train, (i * num_train) + len_train}")
        i += 1
        
        if len(df_train) < len_train:
            break
        
        y = df_train['pred']
        features = [x for x in df_train.columns if x not in ['pred']]
        X = df_train[features]
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # fit models
        # ada.fit(X_train, y_train)
        # bc.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        xt.fit(X_train, y_train)
        
        # get predictions
        # ada_prediction = ada.predict(X_test)
        # bc_prediction = bc.predict(X_test)
        rf_prediction = rf.predict(X_test)
        xt_prediction = xt.predict(X_test)
        
        # ada_accuracy = accuracy_score(y_test.values, ada_prediction)
        # bc_accuracy = accuracy_score(y_test.values, bc_prediction)
        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        xt_accuracy = accuracy_score(y_test.values, xt_prediction)
        
        print(f"RF {rf_accuracy}, XT {xt_accuracy}")
        # ada_RESULTS.append(ada_accuracy)
        # bc_RESULTS.append(bc_accuracy)
        rf_RESULTS.append(rf_accuracy)
        xt_RESULTS.append(xt_accuracy)
    
    # print('ADA Accuracy = ' + str( sum(ada_RESULTS) / len(ada_RESULTS)))
    # print('BC Accuracy = ' + str( sum(bc_RESULTS) / len(bc_RESULTS)))
    print('RF Accuracy = ' + str( sum(rf_RESULTS) / len(rf_RESULTS)))
    print('XT Accuracy = ' + str( sum(xt_RESULTS) / len(xt_RESULTS)))
    return rf, xt
