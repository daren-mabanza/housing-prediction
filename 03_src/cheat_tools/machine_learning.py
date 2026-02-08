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
import shap

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

def resume_metriques_modele(y_train, y_test, y_pred, train_proba, test_proba):
    """
    Évalue un modèle de classification binaire avec métriques complètes
    et affiche uniquement les résultats formatés.
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

    print("=" * 50)
    print("MÉTRIQUES DE PERFORMANCE")
    print("=" * 50)

    for metric, value in metrics.items():
        print(f"{metric:<25} : {value:.4f}")

    print("\n" + "=" * 50)
    print("MATRICE DE CONFUSION")
    print("=" * 50)
    print(f"                     Prédit Négatif | Prédit Positif")
    print(f"Réel Négatif     {cm[0,0]:>14} | {cm[0,1]:>14}")
    print(f"Réel Positif     {cm[1,0]:>14} | {cm[1,1]:>14}")
    print("=" * 50)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSpécificité              : {specificity:.4f}")
    print(f"Faux Positifs            : {fp}")
    print(f"Faux Négatifs            : {fn}")
    print("=" * 50)



# ==============================================================================================
# graphique_courbe_roc
# ==============================================================================================

def graphique_courbe_roc(y_train, train_proba, y_test, test_proba, 
                         figsize=(8, 6), save_path=None):
    """
    Trace les courbes ROC pour les sets d'entraînement et de test
    et affiche les AUC, sans valeur de retour.
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

    overfitting_gap = roc_auc_train - roc_auc_test

    print("\n" + "="*40)
    print("RÉSULTATS ROC-AUC")
    print("="*40)
    print(f"AUC Train    : {roc_auc_train:.4f}")
    print(f"AUC Test     : {roc_auc_test:.4f}")
    print(f"Écart        : {overfitting_gap:.4f}")

    if overfitting_gap > 0.05:
        print("⚠️  Attention : écart élevé (possible overfitting)")
    else:
        print("✓ Écart acceptable")
    print("="*40)



# ==============================================================================================
# graphique_courbe_pr
# ==============================================================================================


def graphique_courbe_pr(y_train, train_proba, y_test, test_proba, 
                        figsize=(8, 6), save_path=None):
    """
    Trace les courbes Précision-Rappel pour train et test
    et affiche les métriques, sans valeur de retour.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score  # [web:323][web:325]

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

    overfitting_gap = avg_precision_train - avg_precision_test
    lift = avg_precision_test / baseline_test if baseline_test > 0 else 0

    print("\n" + "="*40)
    print("RÉSULTATS PRÉCISION-RAPPEL")
    print("="*40)
    print(f"AP Train     : {avg_precision_train:.4f}")
    print(f"AP Test      : {avg_precision_test:.4f}")
    print(f"Écart        : {overfitting_gap:.4f}")
    print(f"Baseline     : {baseline_test:.4f}")
    if overfitting_gap > 0.05:
        print("⚠️  Attention : écart élevé (possible overfitting)")
    else:
        print("✓ Écart acceptable")
    print(f"Lift vs baseline : {lift:.2f}x")
    print("="*40)



# ==============================================================================================
# graphique_courbe_calibration
# ==============================================================================================
import matplotlib.pyplot as plt


def graphique_courbe_calibration(y_train, train_proba, y_test, test_proba, 
                                 n_bins=10, figsize=(8, 6), save_path=None):
    """
    Trace les courbes de calibration pour train et test
    et affiche les métriques de calibration, sans valeur de retour.
    """
    from sklearn.metrics import brier_score_loss  # [web:338][web:344]
    from sklearn.calibration import calibration_curve  # [web:339][web:348]

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

    brier_diff = brier_train - brier_test

    print("\n" + "="*45)
    print("RÉSULTATS DE CALIBRATION")
    print("="*45)
    print(f"Brier Score Train : {brier_train:.4f}")
    print(f"Brier Score Test  : {brier_test:.4f}")
    print(f"Écart Brier       : {brier_diff:.4f}")
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



def dependence_plot(shap_values, var_name, interaction_var=None):
    """
    Génère un graphique de dépendance SHAP pour une variable spécifique.

    Ce graphique montre comment la valeur d'une feature influence sa contribution
    au modèle. La dispersion verticale pour une même valeur de X indique la
    présence d'interactions avec d'autres variables.

    Paramètres
    ----------
    shap_values : shap.Explanation
        L'objet d'explication SHAP complet (calculé via explainer(X)).
    var_name : str
        Nom de la variable à analyser sur l'axe X.
    interaction_var : str, optionnel
        Nom de la variable à utiliser pour la coloration (interaction).
        Si None, SHAP sélectionne automatiquement la variable ayant la plus
        forte interaction apparente.

    Exemple
    -------
    >>> dependence_plot(shap_values, "revenu", interaction_var="education")
    """
    plt.figure(figsize=(8, 5))
    
    if interaction_var:
        # Forcer une variable d'interaction spécifique
        shap.plots.scatter(shap_values[:, var_name], color=shap_values[:, interaction_var])
    else:
        # Laisser SHAP trouver la meilleure interaction automatique
        shap.plots.scatter(shap_values[:, var_name], color=shap_values)
        
    plt.title(f"SHAP Dependence Plot : {var_name}")
    plt.show()



def waterfall_plot(shap_values, X, idx, max_display=15):
    """
    Trace un graphique SHAP de type waterfall pour une observation donnée.
    """
    pos = X.index.get_loc(idx)

    # Pas de plt.figure(), pas de plt.show() ici
    shap.plots.waterfall(shap_values[pos], max_display=max_display)  # show=True par défaut [web:432][web:434]




import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_discrimination_proba_logit(
    model,
    df,
    target,
    class_labels=None,
    palette=None,
    title="Densité des probabilités prédites par classe",
    subtitle="Distribution des scores du modèle selon la classe réelle"
):
    """
    Graphique de densité des probabilités prédites, séparé par classe réelle.

    Ce graphique permet d'évaluer visuellement la capacité de discrimination
    d'un modèle de classification binaire : il compare la distribution des
    probabilités prédites pour Y=0 et Y=1 (ou tout autre libellé fourni via
    `class_labels`).

    Paramètres
    ----------
    model : modèle sklearn ou statsmodels
        Modèle déjà entraîné.
        - sklearn : doit implémenter `predict_proba(X)` (on prend la proba de la classe 1).
        - statsmodels : doit implémenter `predict(X)` avec en sortie une probabilité.
    df : pandas.DataFrame
        Données contenant les features et la colonne cible.
    target : str
        Nom de la colonne cible binaire (0/1).
    class_labels : dict ou None
        Mapping des classes vers des labels lisibles.
        Exemple :
            {0: "Classe négative (0)", 1: "Classe positive (1)"}
        Si None -> {0: "Classe 0", 1: "Classe 1"}.
    palette : dict ou None
        Couleurs par classe {0: couleur_0, 1: couleur_1}.
        Exemple :
            {0: "steelblue", 1: "darkorange"}
        Si None -> {0: "steelblue", 1: "darkorange"}.
    title : str
        Titre principal du graphique.
    subtitle : str
        Sous-titre affiché sous le titre principal.

    Exemple
    -------
    >>> class_labels = {0: "Sains (0)", 1: "Risques (1)"}
    >>> palette = {0: "steelblue", 1: "darkorange"}
    >>> plot_discrimination_proba(
    ...     model=logit,
    ...     df=df,
    ...     target="y",
    ...     class_labels=class_labels,
    ...     palette=palette,
    ...     title="Densité des scores du modèle",
    ...     subtitle="Probabilités prédites pour les classes Sains vs Risques"
    ... )
    """
    # Labels par défaut
    if class_labels is None:
        class_labels = {0: "Classe 0", 1: "Classe 1"}
    if palette is None:
        palette = {0: "steelblue", 1: "darkorange"}

    # 1. Probabilités prédites
    features = df.drop(columns=[target])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    else:
        probs = model.predict(features)

    # 2. Préparation des données
    plot_df = pd.DataFrame({
        "Probabilité": probs,
        "Classe Réelle": df[target].map(class_labels)
    })

    # 3. Graphique
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")

    sns.kdeplot(
        data=plot_df,
        x="Probabilité",
        hue="Classe Réelle",
        fill=True,
        common_norm=False,
        palette={class_labels[0]: palette[0],
                 class_labels[1]: palette[1]},
        alpha=0.2,
        linewidth=2.5
    )

    plt.suptitle(title, fontsize=13, x=0.01, ha="left", color="#2c3e50")
    plt.title(subtitle, fontsize=10, x=0.01, ha="left", color="#555", pad=18)

    plt.xlabel("Probabilité prédite")
    plt.ylabel("Densité")
    plt.xlim(0, 1)
    sns.despine(left=True)
    plt.legend(title="", frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.1), ncol=2)
    plt.tight_layout()
    plt.show()




import numpy as np
import pandas as pd
import plotly.graph_objects as go

def effect_plot_logit(
    model,
    df,
    var,
    target=None,
    at="mean",
    n_points=100,
    class_idx=1,
    title=None,
    xaxis_title=None,
    yaxis_title="Probabilité prédite",
    pain_point=None,
    pain_point_label=None
):
    """
    Trace un effect plot interactif Plotly pour une variable continue.

    La fonction :
      - fait varier `var` sur une grille,
      - fixe les autres variables à un profil de référence (mean/median),
      - prédit la probabilité avec le modèle,
      - trace la courbe Pr(Y=classe_idx | var).

    Paramètres
    ----------
    model : modèle sklearn ou statsmodels
        - sklearn (LogisticRegression, etc.) : doit implémenter `predict_proba(X)`.
        - statsmodels (LogitResults)        : doit implémenter `predict(X)`.
    df : pandas.DataFrame
        Données contenant les features (et éventuellement la cible).
    var : str
        Nom de la variable continue pour laquelle on veut la courbe effet.
    target : str ou None
        Nom de la colonne cible dans df. Si non None, elle est retirée des features.
    at : {"mean", "median"}
        Comment fixer les autres covariables (profil de référence).
    n_points : int
        Nombre de points dans la grille de `var`.
    class_idx : int
        Index de la classe dont on trace la probabilité (1 par défaut).
        Ignoré pour statsmodels (qui renvoie directement Pr(Y=1)).
    title : str ou None
        Titre du graphique. Si None, un titre générique est construit.
    xaxis_title : str ou None
        Label de l'axe X. Si None, utilise `var`.
    yaxis_title : str
        Label de l'axe Y.
    pain_point : float ou None
        Valeur de `var` où tracer une ligne verticale optionnelle.
    pain_point_label : str ou None
        Texte d'annotation au niveau du `pain_point`.

    Retour
    ------
    fig : plotly.graph_objects.Figure
        Figure Plotly interactive à afficher via `fig.show()`.
    """
    # 1. Préparation des features
    X = df.copy()
    if target is not None and target in X.columns:
        X = X.drop(columns=[target])

    if var not in X.columns:
        raise ValueError(f"La variable '{var}' n'est pas présente dans df.")

    # 2. Grille sur var
    x_grid = np.linspace(X[var].min(), X[var].max(), n_points)
    new_df = pd.DataFrame({var: x_grid})

    # 3. Fixer les autres variables au profil de référence
    for col in X.columns:
        if col == var:
            continue
        if X[col].dtype.kind in "bifc":  # numériques
            if at == "mean":
                new_df[col] = X[col].mean()
            elif at == "median":
                new_df[col] = X[col].median()
            else:
                raise ValueError("at doit être 'mean' ou 'median'.")
        else:  # catégorielles
            new_df[col] = X[col].mode()[0]

    # 4. Prédiction des probabilités
    if hasattr(model, "predict_proba"):  # sklearn
        proba = model.predict_proba(new_df)[:, class_idx]
    else:  # statsmodels logit -> probas pour Y=1
        proba = model.predict(new_df)

    # 5. Construction de la figure Plotly
    if xaxis_title is None:
        xaxis_title = var
    if title is None:
        title = f"Effect Plot : évolution de la probabilité selon {var}"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_grid,
        y=proba,
        mode="lines+markers",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=4),
        name=f"Effet de {var}"
    ))

    # Ligne verticale optionnelle
    if pain_point is not None:
        fig.add_vline(
            x=pain_point,
            line_width=2,
            line_dash="dash",
            line_color="#7f8c8d"
        )
        if pain_point_label is not None:
            fig.add_annotation(
                x=pain_point,
                y=max(proba),
                text=pain_point_label,
                showarrow=False,
                yshift=20,
                font=dict(color="#c0392b", size=11)
            )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_tickformat=".0%",
        template="simple_white"
    )

    return fig




