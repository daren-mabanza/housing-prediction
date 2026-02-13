import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def analyser_dependance_khi2(df, target_col, features_cols, alpha=0.05):
    """
    Effectue une série de tests du Khi-deux pour tester l'indépendance entre une variable cible (Y)
    et une liste de variables explicatives (X).
    
    Args:
        df (pd.DataFrame): Le dataframe.
        target_col (str): La variable cible catégorielle (Y).
        features_cols (list): Liste des colonnes catégorielles à tester (X).
        alpha (float): Seuil de significativité (défaut 5%).
        
    Returns:
        pd.DataFrame: Tableau récapitulatif des résultats trié par force de liaison (V de Cramer).
    """
    resultats = []
    
    print(f"--- Analyse de dépendance (Khi-deux) avec la cible : '{target_col}' ---")
    
    for col in features_cols:
        # 1. Nettoyage rapide (suppression des NA pour le test)
        # Le test ne supporte pas les NaN
        df_clean = df[[target_col, col]].dropna()
        
        if df_clean.empty:
            continue
            
        # 2. Table de contingence
        contingency_table = pd.crosstab(df_clean[target_col], df_clean[col])
        
        # Vérification minimale : il faut au moins 2 lignes et 2 colonnes
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            print(f"⚠️ Ignoré : '{col}' (Pas assez de classes pour le test)")
            continue

        # 3. Test du Khi-deux
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # 4. Calcul du V de Cramer (Force de la relation)
        # Formule : sqrt(chi2 / (n * min(k-1, r-1)))
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        # 5. Interprétation
        dependance = "OUI" if p_value < alpha else "NON"
        
        resultats.append({
            "Variable": col,
            "P-value": p_value,
            "Dépendance Significative": dependance,
            "V de Cramer": round(cramer_v, 4),
            "Chi2 Stat": round(chi2, 2)
        })

    # Conversion en DataFrame pour affichage propre
    df_res = pd.DataFrame(resultats)
    
    if not df_res.empty:
        # Tri par V de Cramer décroissant (les liens les plus forts en premier)
        df_res = df_res.sort_values(by="V de Cramer", ascending=False).reset_index(drop=True)
        
        # Formatage cosmétique pour la p-value
        df_res["P-value"] = df_res["P-value"].apply(lambda x: "< 0.001" if x < 0.001 else round(x, 4))
        
    return df_res

# --- EXEMPLE D'UTILISATION ---
# variables_a_tester = ['property_type', 'taille_agglomeration', 'dpe_courant']
# resultats = analyser_dependance_khi2(data_housing, target_col='target_binaire', features_cols=variables_a_tester)
# print(resultats)



import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def analyser_dependance_mannwhitney(df, target_col, features_cols, alpha=0.05):
    """
    Effectue une série de tests U de Mann-Whitney pour tester si la distribution d'une variable continue (X)
    est significativement différente selon les 2 groupes de la target binaire (Y).
    
    Args:
        df (pd.DataFrame): Le dataframe.
        target_col (str): La variable cible binaire (0/1 ou False/True).
        features_cols (list): Liste des variables continues à tester (X).
        alpha (float): Seuil de significativité (défaut 5%).
        
    Returns:
        pd.DataFrame: Tableau récapitulatif trié par taille d'effet (Impact).
    """
    resultats = []
    
    # Vérification que la target est bien binaire
    uniques = df[target_col].dropna().unique()
    if len(uniques) != 2:
        print(f"❌ Erreur : La target '{target_col}' doit avoir exactement 2 classes pour Mann-Whitney. Trouvé : {uniques}")
        return pd.DataFrame()

    group0_label = uniques[0]
    group1_label = uniques[1]
    
    print(f"--- Analyse Mann-Whitney (Target Binaire : '{target_col}') ---")
    print(f"--- Comparaison : Groupe '{group0_label}' vs Groupe '{group1_label}' ---")

    for col in features_cols:
        # 1. Nettoyage et Préparation des 2 groupes
        # On ne garde que les lignes où la variable n'est pas NaN
        subset = df[[target_col, col]].dropna()
        
        group0 = subset[subset[target_col] == group0_label][col]
        group1 = subset[subset[target_col] == group1_label][col]
        
        if len(group0) < 5 or len(group1) < 5:
            print(f"⚠️ Ignoré : '{col}' (Pas assez de données dans un des groupes)")
            continue

        # 2. Test U de Mann-Whitney (Two-sided)
        stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
        
        # 3. Calcul de la Taille d'Effet (Rank-Biserial Correlation)
        # Formule : r = 1 - (2U / (n1 * n2))
        # r varie de -1 à 1. Plus c'est loin de 0, plus l'effet est fort.
        n1 = len(group0)
        n2 = len(group1)
        u_total = n1 * n2
        rank_biserial = 1 - (2 * stat) / u_total
        
        # 4. Interprétation
        dependance = "OUI" if p_value < alpha else "NON"
        
        resultats.append({
            "Variable": col,
            "P-value": p_value,
            "Différence Significative": dependance,
            "Taille Effet (r)": round(abs(rank_biserial), 4), # Valeur absolue pour le tri
            "Direction Effet": "Positif" if rank_biserial > 0 else "Négatif",
            "U Stat": stat
        })

    # Conversion en DataFrame
    df_res = pd.DataFrame(resultats)
    
    if not df_res.empty:
        # Tri par Taille d'Effet décroissante (Impact le plus fort en premier)
        df_res = df_res.sort_values(by="Taille Effet (r)", ascending=False).reset_index(drop=True)
        
        # Formatage cosmétique
        df_res["P-value"] = df_res["P-value"].apply(lambda x: "< 0.001" if x < 0.001 else round(x, 4))
        
    return df_res

# --- EXEMPLE D'UTILISATION ---
# vars_continues = ['size', 'land_size', 'population', 'revenu_median']
# res_mw = analyser_dependance_mannwhitney(data_housing, target_col='target_binaire', features_cols=vars_continues)
# print(res_mw)
