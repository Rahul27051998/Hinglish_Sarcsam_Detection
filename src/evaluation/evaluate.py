import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def evaluate(models,X,y):

    os.makedirs("results/confusion_matrices",exist_ok=True)

    results=[]

    for name,m in models.items():
        pred=m.predict(X)

        acc=accuracy_score(y,pred)
        prec=precision_score(y,pred,average='weighted',zero_division=0)
        rec=recall_score(y,pred,average='weighted',zero_division=0)
        f1=f1_score(y,pred,average='weighted',zero_division=0)

        results.append([name,acc,prec,rec,f1])

        # ===============================
        # UPDATED CONFUSION MATRIX
        # ===============================
        cm = confusion_matrix(y, pred)

        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()

            labels = [
                [f"TN\n{tn}", f"FP\n{fp}"],
                [f"FN\n{fn}", f"TP\n{tp}"]
            ]
        else:
            labels = [["", ""], ["", ""]]

        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap='Blues')
        plt.title(name)

        # Annotate matrix cells
        for i in range(len(labels)):
            for j in range(len(labels[0])):
                plt.text(j, i, labels[i][j],
                         ha='center', va='center', color='black', fontsize=12)

        plt.xticks([0,1], ["Pred 0", "Pred 1"])
        plt.yticks([0,1], ["Actual 0", "Actual 1"])

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.colorbar()
        plt.tight_layout()

        plt.savefig(f"results/confusion_matrices/{name}.png")
        plt.close()
        # ===============================

    df=pd.DataFrame(results,columns=["Model","Accuracy","Precision","Recall","F1_score"])
    df.to_csv("results/test_metrics.csv",index=False)
