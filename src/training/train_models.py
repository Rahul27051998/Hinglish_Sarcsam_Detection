
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from src.models.all_models import get_models

def train_models(X_train,y_train,X_test,y_test):
    models=get_models()
    best_model=None
    best_acc=0
    best_name=None

    os.makedirs("results/plots",exist_ok=True)

    for name,model in models.items():
        train_accs=[]
        test_accs=[]
        for epoch in tqdm(range(20),desc=name):
            model.fit(X_train,y_train)
            train_acc=accuracy_score(y_train,model.predict(X_train))
            test_acc=accuracy_score(y_test,model.predict(X_test))
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        plt.figure()
        plt.plot(train_accs,label="Train")
        plt.plot(test_accs,label="Test")
        plt.legend()
        plt.savefig(f"results/plots/{name}.png")
        plt.close()

        if test_accs[-1]>best_acc:
            best_acc=test_accs[-1]
            best_model=model
            best_name=name

    return models,best_model,best_name
