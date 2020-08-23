import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
import seaborn as sns


def load_data():
    data_2014 = pd.read_csv("data/2014_Financial_Data.csv")
    data_2015 = pd.read_csv("data/2015_Financial_Data.csv")
    data_2016 = pd.read_csv("data/2016_Financial_Data.csv")
    data_2017 = pd.read_csv("data/2017_Financial_Data.csv")
    data_2018 = pd.read_csv("data/2018_Financial_Data.csv")
    data_2018.keys()
    return {2014: data_2014, 2015: data_2015, 2016: data_2016, 2017: data_2017, 2018: data_2018}


def transform_data(datasets, normalize=True, class_mean=True):
    if normalize == True:
        scaler = Normalizer()
    else:
        scaler = StandardScaler()

    for year in datasets.keys():
        #Gi otstranuvame prvite dve koloni bidejki se prazni, a tretata bidejki ne vrshime regresija na datotekata
        datasets[year] = datasets[year].drop(
            columns=["Unnamed: 0","operatingCycle", "cashConversionCycle", "{} PRICE VAR [%]".format(year + 1)])


        #ako class_mean e True, gi popolnuvame praznite polinja vo redicata so prosekot na klasata vo koja pripagja
        #vo taa kolona
        if class_mean == True:
            Class_0 = datasets[year][datasets[year].Class == 0]
            Class_1 = datasets[year][datasets[year].Class == 1]
            for index, row in datasets[year].iterrows():
                for col, value in row.iteritems():
                    if pd.isna(value):
                        if row["Class"] == 0:
                            mean = Class_0[col].mean()
                        else:
                            mean = Class_1[col].mean()
                        datasets[year].at[index, col] = mean

        for column in datasets[year].columns:
            if datasets[year][column].dtype == np.float64:
                
                if class_mean == False:
                    datasets[year][column] = datasets[year][column].fillna(datasets[year][column].mean())
                features = scaler.fit_transform(datasets[year][column].values.reshape(-1, 1))
                datasets[year][column] = features

        lab = LabelEncoder()
        datasets[year]["Sector"] = lab.fit_transform(datasets[year]["Sector"])



def getTrainTestDatasets(datasets, train_keys, test_keys):
    train_datasets = []
    test_datasets = []
    for key in train_keys:
        train_datasets.append(datasets[key])

    for key in test_keys:
        test_datasets.append(datasets[key])
    train = pd.concat(train_datasets)
    test = pd.concat(test_datasets)
    return train, test


def predict_data(X_train, X_test, Y_train, Y_test):
    C_parameters = [0.5, 1, 1.5, 2, 3]
    depth_parameters = [10, 30, 50, 80, 100]
    nn_parameters = [3, 5, 7, 9, 12]
    ann_parameters = [5, 10, 15, 20, 30]
    ann2_parameters = [[5, 5], [5, 10], [10, 5], [10, 10], [15, 15]]
    SVM_results = []
    KNN_results = []
    LR_results = []
    RF_results = []
    ANN_results = []
    ANN2_results = []
    for C, depth, nn, layers, layers_2 in zip(C_parameters, depth_parameters, nn_parameters, ann_parameters,
                                              ann2_parameters):
        svc = SVC(C=C, random_state=0)

        lr = LogisticRegression(C=C, max_iter=1000, random_state=0)

        rf = RandomForestClassifier(max_depth=depth, random_state=0)

        mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=1000, random_state=0)

        mlp2 = MLPClassifier(hidden_layer_sizes=layers_2, max_iter=1000, random_state=0)

        knn = KNeighborsClassifier(n_neighbors=nn, n_jobs=-1)

        svc.fit(X_train, Y_train)
        print("Finished training Support Vector Classifier with C =", C)
        lr.fit(X_train, Y_train)
        print("Finished training Logistic Regression Classifier with C =", C)
        mlp.fit(X_train, Y_train)
        print("Finished training MLP Classifier with single hidden layer perceptron size =", layers)
        mlp2.fit(X_train, Y_train)
        print("Finished training MLP Classifier with two hidden layer perceptron size =", layers_2)
        knn.fit(X_train, Y_train)
        print("Finished training KNN Classifier with number of nearest neighbours =", nn)
        rf.fit(X_train, Y_train)
        print("Finished training Random Forest Classifier with depth =", depth)

        predict_svc = svc.predict(X_test)
        predict_lr = lr.predict(X_test)
        predict_mlp = mlp.predict(X_test)
        predict_mlp2 = mlp2.predict(X_test)
        predict_knn = knn.predict(X_test)
        predict_rf = rf.predict(X_test)

        SVM_results.append(f1_score(Y_test.values.ravel(), predict_svc))
        LR_results.append(f1_score(Y_test.values.ravel(), predict_lr))
        ANN_results.append(f1_score(Y_test.values.ravel(), predict_mlp))
        ANN2_results.append(f1_score(Y_test.values.ravel(), predict_mlp2))
        KNN_results.append(f1_score(Y_test.values.ravel(), predict_knn))
        RF_results.append(f1_score(Y_test.values.ravel(), predict_rf))

    print("Support Vector:", SVM_results)
    print("Logistic Regression:", LR_results)
    print("MLP with 1 hidden layer:", ANN_results)
    print("MLP with 2 hidden layers:", ANN2_results)
    print("K nearest neighbours:", KNN_results)
    print("Random Forest with 100 estimators:", RF_results)
    svmline, = plt.plot(np.arange(0, 5), SVM_results, color="r")
    lrline, = plt.plot(np.arange(0, 5), LR_results, color="g")
    mlpline, = plt.plot(np.arange(0, 5), ANN_results, color="b")
    mlp2line, = plt.plot(np.arange(0, 5), ANN2_results, color="c")
    knnline, = plt.plot(np.arange(0, 5), KNN_results, color="m")
    rfline, = plt.plot(np.arange(0, 5), RF_results, color="y")

    plt.legend([svmline, lrline, mlpline, mlp2line, knnline, rfline],
               ["SVM", "Logistic Regression", "MLP 1 layer", "MLP 2 layers", "KNN", "Random Forest"], loc='right', bbox_to_anchor=(1.3, 0.5))
    plt.show()

def runPredictionWithParameters(normalize=True,class_mean=True):
    print("Prediction with normalize=",normalize,", class_mean=",class_mean)
    print("#"*20)
    print("#"*20)

    # Vchituvanje na datotekite kako Dict(), kade key-value parot e godinata(int) i Pandas Dataframe od datotekata za
    # taa godina
    dataset = load_data()
    full_dataset = pd.concat(dataset.values())
    full_dataset.info()


    # Vo pretposlednata kolona se srekjavame so sluchaj na Price Var [%] za narednata godina (potrebna informacija za
    # ispushtanje na ovaa kolona bidejki ne pravime regresija)
    print(dataset[2014].columns[-2], dataset[2015].columns[-2], dataset[2016].columns[-2], dataset[2017].columns[-2],
          dataset[2018].columns[-2])

    # Vrshime pretprocesiranje na podatocite koristejki Standard Scaler i mean na celata kolona
    transform_data(dataset, normalize=normalize, class_mean=class_mean)
    print("Finished preprocessing data.")

    # Vrshime reduciranje na brojot na koloni so 95% zadrzhen procent na informaciite
    pca = PCA(.95)

    # Se vrshi spojuvanje na site datoteki i se otstranuva Class kolonata
    pca.fit(pd.concat(dataset.values()).drop(columns=["Class"]))
    print("Number of features after PCA:", pca.n_components_)

    #
    #
    # Klasifikacija so trening set od 2014-2016, a test set 2017-2018
    #
    #

    print("Classification of train set (2014-2016) and test set (2017,2018):\n")
    trainSet, testSet = getTrainTestDatasets(dataset, [2014, 2015, 2016], [2017, 2018])

    X_train = trainSet.drop(columns=["Class"])
    Y_train = trainSet["Class"]
    X_test = testSet.drop(columns=["Class"])
    Y_test = testSet["Class"]


    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    predict_data(X_train, X_test, Y_train, Y_test)

    print()
    print()

    print("Classification of train set (2014-2017) and test set 2018:\n")
    trainSet, testSet = getTrainTestDatasets(dataset, [2014, 2015, 2016, 2017], [2018])

    X_train = trainSet.drop(columns=["Class"])
    Y_train = trainSet["Class"]
    X_test = testSet.drop(columns=["Class"])
    Y_test = testSet["Class"]

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    predict_data(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
   #Go vrshime procesot na chitanje, pretprocesiranje, obuka i predviduvanje so razlichni parametri pri pretprocesiranje
   runPredictionWithParameters(normalize=False,class_mean=False)
   runPredictionWithParameters(normalize=False, class_mean=True)
   runPredictionWithParameters(normalize=True, class_mean=False)
   runPredictionWithParameters(normalize=True, class_mean=True)