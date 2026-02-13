from pathlib import Path

import pandas as pd
import numpy as np
from cheat_tools.data_manipulation import multi_astype

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn import set_config

from cheat_tools.machine_learning import resume_metriques_modele, graphique_courbe_roc, graphique_courbe_pr, graphique_courbe_calibration


import joblib


ROOT = Path.cwd().parents[0]
RAW_DATA = ROOT / "01_data" / "01_raw"
PROCESSED_DATA = ROOT / "01_data" / "02_processed"



def pipeline_model():

    set_config(transform_output="pandas")

    # Import des données
    data_housing = pd.read_csv(PROCESSED_DATA / "03_econometrics" / "housing_data_econometrics_processed.csv",
                               dtype={'id_annonce':"object"})

    df = data_housing.copy()

    print("Import des données : OK")
    print("="*50)

    # Nettoyage des variables catégorielles et modification des types
    df["taille_agglomeration"] = df["taille_agglomeration"].fillna("missing")
    df["nb_rooms_bins"] = df["nb_rooms_bins"].fillna("missing")

    multi_astype(
    df,
    ["property_type","taille_agglomeration","nb_rooms_bins"],
    "string"
    )


    # Split Train/Test
    X = df.drop(columns=["target","id_annonce","code_insee"])  
    Y = df["target"]                 


    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,          
        random_state=123,       
        stratify=Y
    )

    print("Split Train/Test : OK")
    print("="*50)


    # Selection du modèle et des hyperparamètres
        # Modèle (Régression logistique avec pénalisation L2)
    logit = LogisticRegression(
        penalty="l2",        
        solver="lbfgs",      
        fit_intercept=True,
        max_iter=2000,
        class_weight=None,   
        random_state=42
    )

        # Hyperparamètres (Equilibré)
    param_logit = {
        "model__C": loguniform(1e-4, 1e2),   
    }


    print("Selection du modèle et des hyperparamètres : OK")
    print("="*50)


    # Création de la Pipeline | Tuning des hyperparamètres | Entrainement du meilleur modèle

        # Création de la pipeline

            # Régression XGBoost pour l'imputation des valeurs manquantes 
    xgb=XGBRegressor(
        n_jobs=-1,
        random_state=42,
        tree_method="hist"
    )

    num_pipeline = Pipeline([
    # Imputation des valeurs numériques manquantes via regression XGBoost
    ("xgb_imputer",IterativeImputer(estimator=xgb,max_iter=3,random_state=42)),

    # Standardisation des variables numériques
    ("scaler",StandardScaler())
    ])


    cat_pipeline = Pipeline([
    # One Hot Encoding  sur les variables catégorielles
    ("ohe",OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'))
    ])


    processing = ColumnTransformer([
    # Pipeline pour les features numériques
    ("num",num_pipeline,make_column_selector(dtype_include=np.number)),

    # Pipeline pour les features catégorielles
    ("cat",cat_pipeline,make_column_selector(dtype_exclude=np.number))
    ])


        # Assemblage de la Pipeline
    pipeline_complete = Pipeline([
    # Processing
    ("processing", processing),

    # Modèle de régression logistique (Ridge)
    ('model', logit)
    ])


    print("Création de la pipeline : OK")
    print("="*50)


        # Tuning des hyperparamètres & Entrainement du meilleur modèle
    recherche = RandomizedSearchCV(
        estimator=pipeline_complete,
        param_distributions=param_logit,
        n_iter = 6,
        scoring = {
            "accuracy":"accuracy",
            "f1":"f1",
            "precision":"precision",
            "recall":"recall",
            "roc_auc":"roc_auc",
            "average_precision":"average_precision",
            "neg_brier_score":"neg_brier_score"
            },
    cv = 3,     
    return_train_score = True,
    refit = "roc_auc",
    random_state = 123,
    n_jobs = -1,
    verbose = 3,
    error_score='raise'
    )

    _ = recherche.fit(X_train, Y_train)

    model = recherche.best_estimator_

    print("Tuning des hyperparamètres et entrainement du meilleur modèle : OK")
    print("="*50)

    # Sauvegarde du modèle (Joblib)
    joblib.dump(model, ROOT / "04_model" / "housing_logit_model.joblib")

    print("Sauvegarde du modèle (Joblib) : OK")
    print("="*50)

    # Sauvegarde de X_train
    X_train.to_csv(PROCESSED_DATA / "04_machine_learning" / "X_train.csv", index=False)

    print("Sauvegarde de X_train : OK")
    print("="*50)

    # Récupérations des probabilités et des prédictions
    train_proba = model.predict_proba(X_train)[:,1]
    train_pred = model.predict(X_train)

    test_proba = model.predict_proba(X_test)[:,1]
    test_pred = model.predict(X_test)


    print("Récupérations des probabilités et des prédictions : OK")
    print("="*50)

    # Résumé des métriques du modèle
    resume_metriques_modele(
        Y_train,
        Y_test,
        test_pred,
        train_proba,
        test_proba
    )

    # Courbe ROC
    graphique_courbe_roc(
    Y_train,train_proba,
    Y_test,test_proba
    )

    # Courbe PR
    graphique_courbe_pr(
    Y_train,train_proba,
    Y_test,test_proba
    )

    # Courbe de calibration
    graphique_courbe_calibration(
    Y_train,train_proba,
    Y_test,test_proba
    )


    print("Résumé des performances du modèle : OK")
    print("="*50)






