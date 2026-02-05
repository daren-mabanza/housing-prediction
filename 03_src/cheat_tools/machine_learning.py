# ==================================
# Nom des fonctions
# ==================================
# resume_metriques_modele : Évalue un modèle de classification binaire avec métriques complètes.
# graphique_courbe_roc : Trace les courbes ROC pour les sets d'entraînement et de test.
# graphique_courbe_pr : Trace les courbes Précision-Rappel pour les sets d'entraînement et de test.
# graphique_courbe_calibration : Trace les courbes de calibration pour les sets d'entraînement et de test.
# GroupImputer : Impute les valeurs manquantes d'une variable numérique en utilisant la médiane (ou moyenne) calculée par groupe sur une autre variable catégorielle.
# ==================================
# Packages nécéssaires 
# ==================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    average_precision_score, roc_auc_score, 
    brier_score_loss, confusion_matrix, f1_score, roc_curve, auc
    )
# ==================================


# ==============================================================================================
# resume_metriques_modele
# ==============================================================================================

def resume_metriques_modele(y_train, y_test, y_pred, train_proba, test_proba, 
                            display_results=True):
    """
    Évalue un modèle de classification binaire avec métriques complètes.
    
    Parameters
    ----------
    y_train : array-like
        Vraies étiquettes du set d'entraînement
    y_test : array-like
        Vraies étiquettes du set de test
    y_pred : array-like
        Prédictions binaires sur le set de test
    train_proba : array-like
        Probabilités prédites sur le set d'entraînement
    test_proba : array-like
        Probabilités prédites sur le set de test
    display_results : bool, default=True
        Si True, affiche les résultats formatés
    
    Returns
    -------
    dict
        Dictionnaire contenant toutes les métriques
    pd.DataFrame
        DataFrame avec les métriques pour export/analyse
    """

    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Average Precision': average_precision_score(y_test, test_proba),
        'ROC-AUC': roc_auc_score(y_test, test_proba),
        'Brier Score (Train)': brier_score_loss(y_train, train_proba),
        'Brier Score (Test)': brier_score_loss(y_test, test_proba)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    df_metrics = pd.DataFrame(list(metrics.items()), 
                              columns=['Metric', 'Value'])
    df_metrics['Value'] = df_metrics['Value'].round(4)
    
    if display_results:
        print("=" * 50)
        print("MÉTRIQUES DE PERFORMANCE")
        print("=" * 50)
        
        for metric, value in metrics.items():
            print(f"{metric:<25} : {value:.4f}")
        
        print("\n" + "=" * 50)
        print("MATRICE DE CONFUSION")
        print("=" * 50)
        print(f"                 Prédit Négatif | Prédit Positif")
        print(f"Réel Négatif     {cm[0,0]:>14} | {cm[0,1]:>14}")
        print(f"Réel Positif     {cm[1,0]:>14} | {cm[1,1]:>14}")
        print("=" * 50)
        
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\nSpécificité              : {specificity:.4f}")
        print(f"Faux Positifs            : {fp}")
        print(f"Faux Négatifs            : {fn}")
        print("=" * 50)
    
    return metrics, df_metrics, cm


# ==============================================================================================
# graphique_courbe_roc
# ==============================================================================================

def graphique_courbe_roc(y_train, train_proba, y_test, test_proba, 
                         figsize=(8, 6), save_path=None):
    """
    Trace les courbes ROC pour les sets d'entraînement et de test.
    
    Parameters
    ----------
    y_train : array-like
        Vraies étiquettes du set d'entraînement
    train_proba : array-like
        Probabilités prédites sur le set d'entraînement
    y_test : array-like
        Vraies étiquettes du set de test
    test_proba : array-like
        Probabilités prédites sur le set de test
    figsize : tuple, default=(8, 6)
        Taille de la figure
    save_path : str, optional
        Chemin pour sauvegarder la figure (ex: 'roc_curve.png')
    
    Returns
    -------
    dict
        Dictionnaire contenant les valeurs AUC pour train et test
    """


    
    fpr_train, tpr_train, _ = roc_curve(y_train, train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, test_proba)
    
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, 
             label=f'Train AUC = {roc_auc_train:.3f}')
    plt.plot(fpr_test, tpr_test, color='red', lw=2, linestyle='--', 
             label=f'Test AUC = {roc_auc_test:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Aléatoire')
    
    plt.xlabel('Taux de faux positifs (FPR)', fontsize=11)
    plt.ylabel('Taux de vrais positifs (TPR)', fontsize=11)
    plt.title('Courbes ROC - Train vs Test', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    
    auc_scores = {
        'AUC_train': roc_auc_train,
        'AUC_test': roc_auc_test,
        'Overfitting': roc_auc_train - roc_auc_test
    }
    
    print("\n" + "="*40)
    print("RÉSULTATS ROC-AUC")
    print("="*40)
    print(f"AUC Train    : {roc_auc_train:.4f}")
    print(f"AUC Test     : {roc_auc_test:.4f}")
    print(f"Écart        : {auc_scores['Overfitting']:.4f}")
    
    if auc_scores['Overfitting'] > 0.05:
        print("⚠️  Attention : écart élevé (possible overfitting)")
    else:
        print("✓ Écart acceptable")
    print("="*40)
    
    return auc_scores


# ==============================================================================================
# graphique_courbe_pr
# ==============================================================================================


def graphique_courbe_pr(y_train, train_proba, y_test, test_proba, 
                        figsize=(8, 6), save_path=None):
    """
    Trace les courbes Précision-Rappel pour les sets d'entraînement et de test.
    
    Parameters
    ----------
    y_train : array-like
        Vraies étiquettes du set d'entraînement
    train_proba : array-like
        Probabilités prédites sur le set d'entraînement
    y_test : array-like
        Vraies étiquettes du set de test
    test_proba : array-like
        Probabilités prédites sur le set de test
    figsize : tuple, default=(8, 6)
        Taille de la figure
    save_path : str, optional
        Chemin pour sauvegarder la figure (ex: 'pr_curve.png')
    
    Returns
    -------
    dict
        Dictionnaire contenant les valeurs AP (Average Precision) pour train et test
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision_train, recall_train, _ = precision_recall_curve(y_train, train_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, test_proba)
    
    avg_precision_train = average_precision_score(y_train, train_proba)
    avg_precision_test = average_precision_score(y_test, test_proba)
    
    baseline_train = y_train.sum() / len(y_train)
    baseline_test = y_test.sum() / len(y_test)
    
    plt.figure(figsize=figsize)
    plt.plot(recall_train, precision_train, color='blue', lw=2, 
             label=f'Train AP = {avg_precision_train:.3f}')
    plt.plot(recall_test, precision_test, color='red', lw=2, linestyle='--', 
             label=f'Test AP = {avg_precision_test:.3f}')
    plt.axhline(y=baseline_test, color='gray', linestyle=':', lw=1, 
                label=f'Baseline = {baseline_test:.3f}')
    
    plt.xlabel('Rappel (Recall)', fontsize=11)
    plt.ylabel('Précision (Precision)', fontsize=11)
    plt.title('Courbes Précision-Rappel - Train vs Test', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    
    ap_scores = {
        'AP_train': avg_precision_train,
        'AP_test': avg_precision_test,
        'Overfitting': avg_precision_train - avg_precision_test,
        'Baseline_train': baseline_train,
        'Baseline_test': baseline_test
    }
    
    print("\n" + "="*40)
    print("RÉSULTATS PRÉCISION-RAPPEL")
    print("="*40)
    print(f"AP Train     : {avg_precision_train:.4f}")
    print(f"AP Test      : {avg_precision_test:.4f}")
    print(f"Écart        : {ap_scores['Overfitting']:.4f}")
    print(f"Baseline     : {baseline_test:.4f}")
    
    if ap_scores['Overfitting'] > 0.05:
        print("⚠️  Attention : écart élevé (possible overfitting)")
    else:
        print("✓ Écart acceptable")
    
    lift = avg_precision_test / baseline_test if baseline_test > 0 else 0
    print(f"Lift vs baseline : {lift:.2f}x")
    print("="*40)
    
    return ap_scores


# ==============================================================================================
# graphique_courbe_calibration
# ==============================================================================================
import matplotlib.pyplot as plt


def graphique_courbe_calibration(y_train, train_proba, y_test, test_proba, 
                                 n_bins=10, figsize=(8, 6), save_path=None):
    """
    Trace les courbes de calibration pour les sets d'entraînement et de test.
    
    Parameters
    ----------
    y_train : array-like
        Vraies étiquettes du set d'entraînement
    train_proba : array-like
        Probabilités prédites sur le set d'entraînement
    y_test : array-like
        Vraies étiquettes du set de test
    test_proba : array-like
        Probabilités prédites sur le set de test
    n_bins : int, default=10
        Nombre de bins pour la courbe de calibration
    figsize : tuple, default=(8, 6)
        Taille de la figure
    save_path : str, optional
        Chemin pour sauvegarder la figure (ex: 'calibration_curve.png')
    
    Returns
    -------
    dict
        Dictionnaire contenant les scores de Brier et l'écart de calibration
    """

    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    
    brier_train = brier_score_loss(y_train, train_proba)
    brier_test = brier_score_loss(y_test, test_proba)
    
    prob_true_train, prob_pred_train = calibration_curve(y_train, train_proba, n_bins=n_bins)
    prob_true_test, prob_pred_test = calibration_curve(y_test, test_proba, n_bins=n_bins)
    
    plt.figure(figsize=figsize)
    plt.plot(prob_pred_train, prob_true_train, marker='o', color='blue', lw=2,
             label=f'Train (Brier={brier_train:.3f})')
    plt.plot(prob_pred_test, prob_true_test, marker='s', color='red', lw=2, linestyle='--',
             label=f'Test (Brier={brier_test:.3f})')
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray', lw=1.5,
             label='Calibration parfaite')
    
    plt.xlabel("Probabilité prédite", fontsize=11)
    plt.ylabel("Probabilité observée", fontsize=11)
    plt.title("Courbes de calibration - Train vs Test", fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")
    
    plt.show()
    
    ece_train = abs(prob_true_train - prob_pred_train).mean()
    ece_test = abs(prob_true_test - prob_pred_test).mean()
    
    calibration_metrics = {
        'Brier_train': brier_train,
        'Brier_test': brier_test,
        'Brier_diff': brier_train - brier_test,
        'ECE_train': ece_train,
        'ECE_test': ece_test
    }
    
    print("\n" + "="*45)
    print("RÉSULTATS DE CALIBRATION")
    print("="*45)
    print(f"Brier Score Train : {brier_train:.4f}")
    print(f"Brier Score Test  : {brier_test:.4f}")
    print(f"Écart Brier       : {calibration_metrics['Brier_diff']:.4f}")
    print(f"\nECE Train         : {ece_train:.4f}")
    print(f"ECE Test          : {ece_test:.4f}")
    
    print("\n" + "-"*45)
    print("INTERPRÉTATION :")
    if brier_test < 0.1:
        print("✓ Excellente calibration (Brier < 0.1)")
    elif brier_test < 0.25:
        print("✓ Bonne calibration (Brier < 0.25)")
    else:
        print("⚠️  Calibration à améliorer (Brier >= 0.25)")
    
    if ece_test < 0.05:
        print("✓ Erreur de calibration faible (ECE < 0.05)")
    elif ece_test < 0.1:
        print("⚠️  Erreur de calibration modérée (ECE < 0.1)")
    else:
        print("⚠️  Erreur de calibration élevée (ECE >= 0.1)")
        print("   → Envisager CalibratedClassifierCV")
    
    print("="*45)
    
    return calibration_metrics



# ==============================================================================================
# GroupImputer
# ==============================================================================================

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Impute les valeurs manquantes d'une variable numérique en utilisant la médiane (ou moyenne)
    calculée par groupe sur une autre variable catégorielle.
    
    Si un groupe rencontré lors du transform() n'était pas présent lors du fit(),
    l'imputation utilise la médiane globale apprise sur le jeu d'entraînement.

        >> GroupImputer(group_by="property_type", variables=["size", "nb_rooms", "nb_bedrooms", "nb_bathrooms"])

    """
    def __init__(self, group_by, variables, strategy='median'):
        """
        Args:
            group_by (str): Nom de la colonne catégorielle (ex: 'property_type').
            variables (list): Liste des colonnes numériques à imputer (ex: ['size', 'nb_rooms']).
            strategy (str): 'median' ou 'mean'.
        """
        self.group_by = group_by
        self.variables = variables
        self.strategy = strategy
        # Mémoire de l'apprentissage
        self.statistics_ = {} 
        self.global_statistics_ = {}
        
    def fit(self, X, y=None):
        # Vérifications de base
        if self.group_by not in X.columns:
            raise ValueError(f"La colonne de groupement '{self.group_by}' est absente.")
            
        for var in self.variables:
            if var not in X.columns:
                raise ValueError(f"La variable '{var}' est absente.")
                
            # 1. Calcul de la statistique globale (pour les cas inconnus)
            if self.strategy == 'median':
                self.global_statistics_[var] = X[var].median()
                self.statistics_[var] = X.groupby(self.group_by)[var].median()
            elif self.strategy == 'mean':
                self.global_statistics_[var] = X[var].mean()
                self.statistics_[var] = X.groupby(self.group_by)[var].mean()
                
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        for var in self.variables:
            # On ne traite que les lignes qui ont réellement besoin d'aide (NaN)
            mask_na = X_copy[var].isna()
            
            if not mask_na.any():
                continue
                
            # 1. On essaie de mapper avec la médiane du groupe
            # Si le groupe est inconnu, map renvoie NaN
            imputed_values = X_copy.loc[mask_na, self.group_by].map(self.statistics_[var])
            
            # 2. On comble les trous restants (groupes inconnus) avec la globale
            imputed_values = imputed_values.fillna(self.global_statistics_[var])
            
            # 3. On injecte les valeurs calculées
            X_copy.loc[mask_na, var] = imputed_values
            
        return X_copy


