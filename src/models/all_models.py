
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

def get_models():
    return {
        "SVM":SVC(probability=True),
        "RandomForest":RandomForestClassifier(),
        "LogisticRegression":LogisticRegression(max_iter=1000),
        "NaiveBayes":MultinomialNB(),
        "DecisionTree":DecisionTreeClassifier(),
        "KNN":KNeighborsClassifier(),
        "GradientBoosting":GradientBoostingClassifier(),
        "AdaBoost":AdaBoostClassifier(),
        "ExtraTrees":ExtraTreesClassifier(),
        "SGD":SGDClassifier()
    }
