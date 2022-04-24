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


def test_user_date(test_data):
    df = pd.read_csv('hepatitis_csv.csv')
    print(df.shape)
    df.head()
    df.info()
    df.isna().mean()
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('float')}
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('int')}
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('object')}
    {columns: len(df[columns].unique()) for columns in df.select_dtypes('bool')}
    X, y = preprocess_inputs(df, drop_protime=True)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_reduced = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=["PC1", "PC2"])
    X_reduced = pd.concat([X_reduced, y, pd.Series(cluster_labels, name='cluster')], axis=1)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    # X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    #X_test = pd.DataFrame(scaler.transform([test_data]), index=X_test.index)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict([test_data])
    return y_pred