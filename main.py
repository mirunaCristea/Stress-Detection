# main_train_all.py
from build_dataset import build_full_dataset
from modele.run_loso import run_loso
from build_dataset import save_feature_dataset, load_feature_dataset
def main():
# pt salvarea de feature-uri, de comentat dupa ce le-am salvat adica dupa ce le am rulat o data
    X, y, groups = build_full_dataset("data/WESAD")
    save_feature_dataset(X, y, groups, "data/features/wesad_features_all.csv")
    #X, y, groups = build_full_dataset("data/WESAD")

    # X, y, groups = load_feature_dataset("data/features/wesad_features_all.csv")
       
       
    # #     # # antrenează
    # # 
    # res_rf = run_loso(X, y, groups, model_name="rf")



    # # #res_logreg = run_loso(X, y, groups, model_name="logreg")


    # # salvezi rezultate
    res_rf.to_csv("data/results/loso_rf.csv", index=False)

if __name__ == "__main__":
    main()
