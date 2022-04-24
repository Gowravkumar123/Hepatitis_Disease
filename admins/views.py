from django.contrib import messages
from django.shortcuts import render
from users.models import UserRegistrationModel


# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/viewregisterusers.html', {'data': data})


def classification_result(request):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_style('darkgrid')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv('hepatitis_csv2.csv')
    print(df.shape)
    df.head()
    df.info()
    df.isna().mean()
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('float')}
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('int')}
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('object')}
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('bool')}

    def preprocess_inputs(df, drop_protime=False):
        df = df.copy()
        # Identify the continuous numeric features
        continuous_features = ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']
        # Fill missing values
        for column in continuous_features:
            df[column] = df[column].fillna(df[column].mean())
        for column in df.columns.drop(continuous_features):
            df[column] = df[column].fillna(df[column].mode().sample(1, random_state=1).values[0])
            # Convert the booleans columns into integer columns
            for column in df.select_dtypes('bool'):
                df[column] = df[column].astype(np.int)
        # Encode the sex column as a binary feature
        df['sex'] = df['sex'].replace({
            'female': 0,
            'male': 1
        })
        df['class'] = df['class'].replace({
            'live': 0,
            'die': 1
        })
        # Shuffle the data
        df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        # Change label name
        df = df.rename(columns={'class': 'label'})
        # Drop protime
        if drop_protime == True:
            df = df.drop('protime', axis=1)
        # Split df into X and y
        y = df['label']
        X = df.drop('label', axis=1)
        return X, y

    X, y = preprocess_inputs(df, drop_protime=True)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    cluster_labels
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_reduced = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=["PC1", "PC2"])
    X_reduced = pd.concat([X_reduced, y, pd.Series(cluster_labels, name='cluster')], axis=1)
    centroids = pca.transform(kmeans.cluster_centers_)
    X_reduced
    kmeans.cluster_centers_
    centroids = pca.transform(kmeans.cluster_centers_)
    Cluster_ex_0 = X_reduced.query('cluster == 0')
    Cluster_ex_1 = X_reduced.query('cluster == 1')
    Cluster_ex_0 = X_reduced.query('cluster == 0')
    Cluster_ex_1 = X_reduced.query('cluster == 1')
    live_examples = X_reduced.query("label == 'live'")
    die_examples = X_reduced.query("label == 'die'")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy1 = accuracy_score(y_test, y_pred) * 100
    print('Accuracy:', accuracy1)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    from sklearn.metrics import precision_score
    precision1 = precision_score(y_test, y_pred) * 100
    print('Precision Score:', precision1)
    from sklearn.metrics import recall_score
    recall1 = recall_score(y_test, y_pred) * 100
    from sklearn.metrics import f1_score
    f1score1 = f1_score(y_test, y_pred) * 100
    roc1 = roc_auc_score(y_test, y_pred) * 100
    model2 = SVC()
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy2 = accuracy_score(y_test, y_pred2) * 100
    print('Accuracy:', accuracy2)
    precision2 = precision_score(y_test, y_pred2) * 100
    recall2 = recall_score(y_test, y_pred2) * 100
    f1score2 = f1_score(y_test, y_pred2) * 100
    roc2 = roc_auc_score(y_test, y_pred2) * 100
    model3 = GaussianNB()
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    accuracy3 = accuracy_score(y_test, y_pred3) * 100
    precision3 = precision_score(y_test, y_pred3) * 100
    recall3 = recall_score(y_test, y_pred3) * 100
    f1score3 = f1_score(y_test, y_pred3) * 100
    roc3 = roc_auc_score(y_test, y_pred3) * 100

    model4 = KNeighborsClassifier()
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    accuracy4 = accuracy_score(y_test, y_pred4) * 100
    precision4 = precision_score(y_test, y_pred4) * 100
    recall4 = recall_score(y_test, y_pred4) * 100
    f1score4 = f1_score(y_test, y_pred4) * 100
    roc4 = roc_auc_score(y_test, y_pred4) * 100
    model5 = MLPClassifier()
    model5.fit(X_train, y_train)
    y_pred5 = model5.predict(X_test)
    accuracy5 = accuracy_score(y_test, y_pred5) * 100
    precision5 = precision_score(y_test, y_pred5) * 100
    recall5 = recall_score(y_test, y_pred5) * 100
    f1score5 = f1_score(y_test, y_pred5) * 100
    roc5 = roc_auc_score(y_test, y_pred5) * 100

    accuracy = {'RF': accuracy1, 'SVM': accuracy2, 'NaiveBayes': accuracy3, 'KNN': accuracy4, 'MLP': accuracy5}
    precision = {'RF': precision1, 'SVM': precision2, 'NaiveBayes': precision3, 'KNN': precision4, 'MLP': precision5}
    recall = {'RF': recall1, 'SVM': recall2, 'NaiveBayes': recall3, 'KNN': recall4, 'MLP': recall5}
    f1score = {'RF': f1score1, 'SVM': f1score2, 'NaiveBayes': f1score3, 'KNN': f1score4, 'MLP': f1score5}
    roc = {'RF': roc1, 'SVM': roc2, 'NaiveBayes': roc3, 'KNN': roc4, 'MLP': roc5}
    return render(request, 'admins/classification_results.html',
                  {"accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score, "roc": roc})
