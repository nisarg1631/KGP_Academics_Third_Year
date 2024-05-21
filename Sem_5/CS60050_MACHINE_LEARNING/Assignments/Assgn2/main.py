from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
from model import NaiveBayes
from math import sqrt

def read_data():
    df = pd.read_csv("train.csv")
    return df


def get_M_matrix():
    nltk.download('stopwords')
    stops = set(stopwords.words('english'))

    vocab = {}
    text_arr = df['text'].to_numpy()
    for i in range(len(text_arr)):
        for word in re.findall("[a-z0-9]+", text_arr[i].casefold()):
            if (word not in stops) and (len(word) > 2):
                vocab[word] = 1

    print(f"Number of distinct words in the dataset = {len(vocab)}\n")
    M = np.zeros((len(text_arr), len(vocab)))

    idx = 0
    for word in vocab:
        vocab[word] = idx
        idx += 1

    for i in tqdm(range(len(text_arr))):
        for word in re.findall("[a-z0-9]+", text_arr[i].casefold()):
            if (word not in stops) and (len(word) > 2):
                M[i][vocab[word]] = 1
    return M


def train_test_split_data(M, y):
    X_train, X_test, y_train, y_test = train_test_split(M, y, stratify=y, test_size=0.30, random_state=42) 
    print(f"Shape X train {X_train.shape}")
    print(f"Shape y train {y_train.shape}")
    print(f"Shape X test {X_test.shape}")
    print(f"Shape y test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def run_experiment(X_train,X_test,y_train,y_test,le,alpha=1):
    model = NaiveBayes(alpha=alpha,n_classes=len(le.classes_))
    model.fit(X_train, y_train)
    y_pred,y_probs = model.predict(X_test)
    
    print(f"Confusion Matrix:\n")
    mat = confusion_matrix(y_test, y_pred)
    print(f"     {le.classes_}")
    for ind, val in enumerate(le.classes_):
        print(f"{val}: {mat[ind]}")

    print(f"\nFull Classification report:")
    # print(classification_report(y_test, y_pred))

    for ind, val in enumerate(le.classes_):
        print(f"\nResults for class {val}:")
        tp = mat[ind][ind]  # current row and col
        tn = np.sum(np.delete(np.delete(mat, ind, 0), ind, 1)) # drop the current row and col
        fp = np.sum(np.delete(mat[:,ind], ind, 0)) # take the current col without the current row
        fn = np.sum(np.delete(mat[ind, :], ind, 0)) # take the current row without the current col
        print(f"True positives: {tp}, True negatives: {tn}, False positives: {fp}, False negatives: {fn}")
        sensitivity = tp/(tp+fn)
        print(f"Sensitivity/Recall: {sensitivity}")
        print(f"Specificity: {tn/(tn+fp)}")
        precission = tp/(tp+fp)
        print(f"Precision: {precission}")
        print(f"F-score: {(2*precission*sensitivity)/(precission+sensitivity)}")


    accuracy = accuracy_score(y_test,y_pred)
    interval = 1.96 * sqrt( (accuracy * (1 - accuracy)) / X_test.shape[0])
    print(f"\nAccuracy on test split = {accuracy}")
    print(f"95% confidence interval of accuracy on test split = {interval}")
    print(f"Hence, accuracy is between {accuracy-interval} and {accuracy+interval}")


if __name__ == "__main__":
    np.seterr(all="ignore")

    df = read_data()
    M = get_M_matrix()
    le = preprocessing.LabelEncoder()
    le.fit(df.author)

    print(f"\nClass mapping:")
    for num, val in enumerate(le.classes_):
        print(f"\t{num} -> {val}")

    print(f"\nTotal samples: {M.shape[0]}")
    print(f"Vocab size: {M.shape[1]}")

    df.author = le.transform(df.author)
    X_train, X_test, y_train, y_test = train_test_split_data(M, df.author.to_numpy())

    print("\n" + "#-"*10 + f"\tExperiment with no laplace correction\t" + "-#"*10)
    run_experiment(X_train,X_test,y_train,y_test,le,alpha=0)
    print("\n" + "#-"*10 + f"\tExperiment with laplace correction, alpha = 1\t" + "-#"*10)
    run_experiment(X_train,X_test,y_train,y_test,le,alpha=1)
    print("\n" + "#-"*10 + f"\tExperiment with laplace correction, alpha = 10\t" + "-#"*10)
    run_experiment(X_train,X_test,y_train,y_test,le,alpha=10)
