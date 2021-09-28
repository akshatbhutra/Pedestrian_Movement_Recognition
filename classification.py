def classifier(leftAngle,rightAngle):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression 
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics 

    df=pd.read_csv("dataset.csv")
    # df.head()

    # feature variables
    x = df.drop(['Outcome'], axis=1)

    # target variable
    y = df.Outcome
    # y

    df1 = df[df['Outcome'] == 1]
    y1  = df1.Outcome
    # y1

    df2 = df[df['Outcome'] == 0]
    y2  = df2.Outcome
    # y2

    df3 = df1.sample(n = 286)
    # df3

    df_final = pd.concat([df2, df3], axis=0)
    # df_final

    df_final = df_final.sample(frac = 1)
    # df_final

    x_final = df_final.drop(['Outcome'], axis=1)
    # x_final

    y_final = df_final.Outcome
    # y_final

    x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2, random_state=1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    model = LogisticRegression()

    model = model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

    confusion_matrix(y_test,y_pred)

    #Evaluation using Classification report
    from sklearn.metrics import classification_report
    # print(classification_report(y_test,y_pred))

    # checking prediction value
    out = model.predict([[leftAngle,rightAngle]])
    # print(out[0])
    
    return out[0]

# classifier()