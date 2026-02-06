import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def plot_grille_distribution(df, columns=None, n_cols=3, figsize=(16, 4)):
    """
    Affiche une grille de graphiques de distribution (histplot + KDE) pour les colonnes numériques.

    Args:
        df (pd.DataFrame): Le dataframe contenant les données.
        columns (list, optional): Liste spécifique de colonnes à tracer. Si None, prend toutes les numériques.
        n_cols (int): Nombre de graphiques par ligne.
        figsize (tuple): Taille de base pour UNE ligne de graphiques. La hauteur totale s'adapte.
    """
    # 1. Sélection des colonnes numériques si non spécifiées
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    n_vars = len(columns)
    if n_vars == 0:
        print("Aucune colonne numérique à afficher.")
        return

    # 2. Calcul des dimensions de la grille
    n_rows = math.ceil(n_vars / n_cols)
    
    # Ajustement de la hauteur totale de la figure
    total_figsize = (figsize[0], figsize[1] * n_rows)
    
    # 3. Création de la figure et des axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize, constrained_layout=True)
    
    # Aplatir les axes pour une itération facile (gère le cas 1 seule ligne ou 1 seul plot)
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 4. Boucle de traçage
    for i, col in enumerate(columns):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution : {col}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('') # Nettoyer pour plus de clarté
        
    # 5. Masquer les axes vides restants (si le nombre de plots n'est pas un multiple de n_cols)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()

# --- Exemple d'utilisation ---
# plot_distributions_grid(df)
# plot_distributions_grid(df, columns=['age', 'salaire', 'score'], n_cols=2)



def plot_boxplots_grid(df, columns=None, n_cols=3, figsize=(16, 4)):
    """
    Affiche une grille de Boxplots pour les colonnes numériques.
    Idéal pour visualiser les médianes, quartiles et outliers.

    Args:
        df (pd.DataFrame): Le dataframe.
        columns (list, optional): Liste des colonnes. Sinon, auto-détection.
        n_cols (int): Nombre de plots par ligne.
        figsize (tuple): Taille de base pour UNE ligne.
    """
    # 1. Auto-détection des numériques
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    n_vars = len(columns)
    if n_vars == 0:
        print("Aucune colonne numérique trouvée.")
        return

    # 2. Calcul de la grille
    n_rows = math.ceil(n_vars / n_cols)
    total_figsize = (figsize[0], figsize[1] * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize, constrained_layout=True)
    
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 3. Boucle de traçage
    for i, col in enumerate(columns):
        # x=df[col] pour un boxplot horizontal (plus lisible pour les noms d'axes)
        # y=df[col] pour vertical
        sns.boxplot(x=df[col], ax=axes[i], color='lightgreen')
        
        axes[i].set_title(f'Boxplot : {col}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('') # On épure
        
    # 4. Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()

# --- Utilisation ---
# plot_boxplots_grid(df)


import textwrap

def plot_countplots_grid(df, columns=None, n_cols=3, base_height=4, max_label_length=15):
    """
    Affiche une grille de countplots sans warnings (compatible Seaborn v0.14+).
    """
    # 1. Auto-détection
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    n_vars = len(columns)
    if n_vars == 0:
        print("Aucune colonne catégorielle trouvée.")
        return

    # 2. Calcul de la grille
    n_rows = math.ceil(n_vars / n_cols)
    total_figsize = (5 * n_cols, base_height * n_rows) 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize, constrained_layout=True)
    
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 3. Boucle de traçage
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # On récupère l'ordre pour fixer les axes
        value_counts = df[col].value_counts()
        order = value_counts.index
        n_categories = len(order)
        
        # --- CORRECTION 1 (Seaborn) ---
        # On map 'hue' sur la même variable que 'x' et on désactive la légende
        sns.countplot(
            data=df, 
            x=col, 
            ax=ax, 
            order=order, 
            palette='viridis', 
            hue=col,       # Assignation explicite
            legend=False   # Pas de légende redondante
        )
        
        # --- Gestion des labels ---
        labels = [str(x) for x in order]
        max_len_txt = max([len(l) for l in labels]) if labels else 0
        
        # --- CORRECTION 2 (Matplotlib) ---
        # On fixe d'abord les positions des ticks avant de toucher aux labels
        ax.set_xticks(range(len(labels)))

        # Wrapping du texte si nécessaire
        if max_len_txt > max_label_length:
            labels = [textwrap.fill(l, width=max_label_length) for l in labels]
        
        # Application des labels (maintenant que les ticks sont fixés)
        ax.set_xticklabels(labels)

        # --- Logique de Rotation ---
        if n_categories > 10 or max_len_txt > 20:
            rotation = 90
            ha = 'center'
        elif n_categories > 5 or max_len_txt > 10:
            rotation = 45
            ha = 'right'
        else:
            rotation = 0
            ha = 'center'

        ax.tick_params(axis='x', rotation=rotation)
        
        # Ajustement fin pour l'alignement oblique
        if rotation == 45:
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')

        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Effectif')

    # 4. Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()




def afficher_heatmap_correlation(df, colonnes=None, titre='Matrice de Corrélation'):
    """
    Affiche une heatmap de corrélation triangulaire optimisée.
    
    Args:
        df (pd.DataFrame): Le dataframe contenant les données.
        colonnes (list, optional): Liste des colonnes à inclure. Par défaut: toutes les numériques.
        titre (str): Le titre du graphique.
    """
    # 1. Filtrage des données
    if colonnes:
        data = df[colonnes]
    else:
        # Sélection automatique des colonnes numériques si aucune liste n'est fournie
        data = df.select_dtypes(include=[np.number])

    # 2. Calcul de la corrélation
    corr = data.corr()

    # 3. Création du masque "Triangle" (cache la partie supérieure redondante)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 4. Calcul dynamique de la taille de la figure
    # On compte les variables pour adapter la taille (environ 0.8 pouce par variable)
    n_vars = len(data.columns)
    taille_ref = max(10, n_vars * 0.8)  # Minimum 10x10, sinon proportionnel
    
    plt.figure(figsize=(taille_ref, taille_ref * 0.8)) # Ratio légèrement rectangulaire pour le titre

    # 5. Configuration de la Heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',       # Palette divergente (Bleu <-> Rouge)
        vmax=1, vmin=-1, center=0, # Bornes fixes pour une lecture correcte des couleurs
        square=True,           # Force les cellules à être carrées
        linewidths=.5,         # Séparateur blanc propre
        cbar_kws={"shrink": .75}, # Légende latérale ajustée
        annot=True,            # Affiche les coefficients
        fmt=".2f",             # Format à 2 décimales (plus lisible que le défaut)
        annot_kws={"size": 10 if n_vars < 15 else 8} # Réduit la police si beaucoup de variables
    )

    # 6. Esthétique finale
    plt.title(titre, fontsize=16, weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right') # Rotation pour éviter le chevauchement
    plt.yticks(rotation=0)
    
    plt.show()





def smart_countplot(df, colonne, titre=None, xlabel=None, ylabel="Nombre d'observations", rotation=45, figsize=(12, 6)):
    """
    Affiche un countplot trié et lisible pour une variable catégorielle.
    
    Args:
        df (pd.DataFrame): Le dataframe.
        colonne (str): Le nom de la colonne à analyser.
        titre (str, optional): Titre du graphique. Défaut: "Répartition : {colonne}".
        xlabel (str, optional): Label axe X. Défaut: Nom de la colonne.
        ylabel (str, optional): Label axe Y. Défaut: "Nombre d'observations".
        rotation (int): Angle de rotation des labels (45 ou 90 souvent).
        figsize (tuple): Taille de la figure.
    """
    
    # 1. Gestion des valeurs par défaut pour les titres
    if titre is None:
        titre = f"Répartition : {colonne}"
    if xlabel is None:
        xlabel = colonne

    plt.figure(figsize=figsize)

    # 2. Calcul de l'ordre décroissant (Pareto)
    try:
        ordre = df[colonne].value_counts().index
    except KeyError:
        print(f"Erreur : La colonne '{colonne}' n'existe pas dans le dataframe.")
        return

    # 3. Création du graphique
    sns.countplot(
        data=df, 
        x=colonne, 
        order=ordre,          
        palette="viridis",    
        hue=colonne,          # Évite le FutureWarning de Seaborn
        legend=False          
    )

    # 4. Esthétique et Labels
    plt.title(titre, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Rotation intelligente avec alignement à droite pour la lisibilité
    plt.xticks(rotation=rotation, ha='right')

    plt.tight_layout()
    plt.show()




import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def plot_scatter_matrix_dynamique(df, cols=None, max_vars=15):
    """
    Affiche une scatter matrix optimisée et lisible.
    Gère automatiquement la taille de la figure et la densité des points.
    
    Args:
        df (pd.DataFrame): Le dataframe.
        cols (list): Liste des colonnes à inclure. Si None, prend les numériques.
        max_vars (int): Limite de sécurité pour éviter de planter l'affichage (défaut 15).
    """
    # 1. Sélection et Validation des colonnes
    if cols is None:
        cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Sécurité : on coupe si trop de variables
    if len(cols) > max_vars:
        print(f"⚠️ Trop de variables ({len(cols)}). On garde les {max_vars} premières pour la lisibilité.")
        cols = cols[:max_vars]
        
    n_vars = len(cols)
    if n_vars < 2:
        print("Erreur : Il faut au moins 2 variables pour une scatter matrix.")
        return

    # 2. Calcul dynamique de la taille (figsize)
    # Règle : 2.5 pouces par variable, avec un min de 8 et max de 20
    fig_size_unit = min(max(8, n_vars * 2.2), 25)
    
    # 3. Paramètres graphiques adaptés à la densité
    # Plus il y a de lignes, plus les points doivent être petits et transparents
    # Si bcp de données (> 1000 lignes), on allège
    n_rows = len(df)
    alpha_val = 0.5 if n_rows < 1000 else 0.3
    s_val = 20 if n_rows < 1000 else 5  # Taille des points

    # 4. Traçage
    print(f"Génération de la matrice {n_vars}x{n_vars}...")
    
    axes = scatter_matrix(
        df[cols],
        figsize=(fig_size_unit, fig_size_unit),
        alpha=alpha_val,       # Transparence
        diagonal='kde',        # 'kde' (courbe) est souvent plus lisible que 'hist' (barres) sur la diagonale
        s=s_val,               # Taille des points
        color='#1f77b4',       # Bleu standard agréable
        grid=True              # Grille pour repérer les alignements
    )
    
    # 5. Amélioration des labels (Rotation et Taille)
    # On parcourt tous les sous-graphiques pour ajuster les axes
    for ax in axes.flatten():
        # Labels X
        ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=45, ha='right')
        # Labels Y
        ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0, labelpad=20)
        
        # On supprime les ticks vides pour alléger
        ax.yaxis.label.set_visible(True) 
        
    plt.suptitle(f"Scatter Matrix ({n_vars} variables)", y=1.02, fontsize=16, fontweight='bold')
    # plt.tight_layout() # Attention : tight_layout bug parfois avec scatter_matrix, on l'évite ici ou on le gère avec précaution
    
    plt.show()

# --- Exemple ---
# variables_interet = ['price', 'size', 'land_size', 'nb_rooms', 'population']
# plot_scatter_matrix_dynamique(data_housing, variables_interet)



import math

def plot_kde_target_binaire_grid(df, target_col, continuous_vars, n_cols=3, figsize=(18, 5)):
    """
    Affiche une grille de KDE plots pour comparer les distributions selon la target binaire.
    
    Args:
        df (pd.DataFrame): Le dataframe.
        target_col (str): Nom de la colonne cible (0/1).
        continuous_vars (list): Liste des variables continues à analyser.
        n_cols (int): Nombre de graphiques par ligne.
        figsize (tuple): Taille de base pour UNE ligne (largeur, hauteur).
    """
    # 1. Préparation de la grille
    n_vars = len(continuous_vars)
    if n_vars == 0:
        return

    n_rows = math.ceil(n_vars / n_cols)
    
    # Taille totale ajustée au nombre de lignes
    total_figsize = (figsize[0], figsize[1] * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize, constrained_layout=True)
    
    # Aplatir axes pour itérer facilement (si 1 seule variable, ce n'est pas une liste)
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 2. Boucle de traçage
    for i, col in enumerate(continuous_vars):
        ax = axes[i]
        
        # KDE Plot Classe 0
        sns.kdeplot(
            data=df[df[target_col] == 0], 
            x=col, 
            label="Prix < 350k", 
            fill=True, 
            alpha=0.3, 
            ax=ax,
            color='#1f77b4' # Bleu
        )
        
        # KDE Plot Classe 1
        sns.kdeplot(
            data=df[df[target_col] == 1], 
            x=col, 
            label="Prix > 350k", 
            fill=True, 
            alpha=0.3, 
            ax=ax,
            color='#ff7f0e' # Orange
        )
        
        ax.set_title(f"Distibution : {col}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Densité")
        ax.set_xlabel(col)
        
        # Légende unique pour éviter la répétition (ou seulement sur le premier)
        if i == 0:
            ax.legend(loc='upper right', frameon=True)
        else:
            ax.legend().remove() # On enlève pour alléger

    # 3. Masquer les axes vides (s'il y a moins de vars que de cases)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Comparaison des distributions (Target : {target_col})", y=1.02, fontsize=16)
    plt.show()

# --- Exemple d'utilisation ---
# vars_a_tester = ['size', 'land_size', 'nb_rooms', 'population', 'revenu_median', 'proximite_gare']
# plot_kde_target_binaire_grid(data_housing, 'target', vars_a_tester)



def plot_countplots_bivarie_grid(df, target_col, categorical_cols, n_cols=3, figsize=(18, 5)):
    """
    Affiche une grille de Countplots bi-variés (Variable X segmentée par Target).
    Permet de voir si une catégorie est sur-représentée dans la classe cible.
    
    Args:
        df (pd.DataFrame): Le dataframe.
        target_col (str): La variable cible (ex: 'target_binaire').
        categorical_cols (list): Liste des variables catégorielles à analyser.
        n_cols (int): Nombre de graphiques par ligne.
    """
    # 1. Gestion de la grille
    n_vars = len(categorical_cols)
    if n_vars == 0:
        return

    n_rows = math.ceil(n_vars / n_cols)
    total_figsize = (figsize[0], figsize[1] * n_rows) # Hauteur adaptative
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize, constrained_layout=True)
    
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 2. Boucle de traçage
    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        
        # Astuce : On calcule l'ordre global pour que les barres soient triées
        order = df[col].value_counts().index
        
        sns.countplot(
            data=df, 
            x=col, 
            hue=target_col,       # C'est ici que la magie opère (segmentation)
            order=order,          # On garde l'ordre logique
            palette="viridis",    # Couleurs distinctes
            ax=ax
        )
        
        # Titres et Labels
        ax.set_title(f"{col} vs {target_col}", fontsize=11, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Effectif")
        
        # Gestion intelligente de la rotation des labels
        if len(order) > 4 or any(len(str(s)) > 10 for s in order):
            ax.tick_params(axis='x', rotation=45)
            # Alignement à droite pour éviter que le texte chevauche la barre suivante
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        
        # Gestion de la légende (uniquement sur le 1er graph pour ne pas polluer)
        if i == 0:
            ax.legend(title=target_col, loc='upper right')
        else:
            ax.get_legend().remove()

    # 3. Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Analyse Bi-variée Catégorielle (Segmentation par {target_col})", y=1.02, fontsize=16)
    plt.show()

# --- Exemple ---
# vars_cat = ["position_commune_unite_urbaine", "taille_agglomeration", "type_unite_urbaine"]
# plot_countplots_bivarie_grid(data_housing, "target", vars_cat)


import seaborn as sns
import matplotlib.pyplot as plt
import math

def plot_kde_multivariables_grid(df, features_cols, target_col=None, n_cols=3, figsize=(18, 5)):
    """
    Affiche une grille de graphiques de densité (KDE) pour une liste de variables.
    Optionnellement segmenté par une target binaire (style comparatif).
    
    Args:
        df (pd.DataFrame): Le dataframe.
        features_cols (list): Liste des variables continues à tracer.
        target_col (str, optional): Si fourni, sépare les courbes par classe (ex: 'target').
        n_cols (int): Nombre de graphiques par ligne.
    """
    # 1. Gestion de la grille
    n_vars = len(features_cols)
    if n_vars == 0:
        return

    n_rows = math.ceil(n_vars / n_cols)
    total_figsize = (figsize[0], figsize[1] * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize, constrained_layout=True)
    
    # Aplatir axes pour itérer facilement
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 2. Boucle de traçage
    for i, col in enumerate(features_cols):
        ax = axes[i]
        
        if target_col:
            # Mode Comparatif (Bicolore)
            sns.kdeplot(data=df, x=col, hue=target_col, fill=True, alpha=0.3, palette='tab10', common_norm=False, ax=ax)
        else:
            # Mode Simple (Monocolore)
            sns.kdeplot(data=df, x=col, fill=True, alpha=0.5, color='#1f77b4', ax=ax)
            
        ax.set_title(f"Distribution : {col}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Densité")
        ax.set_xlabel("") # On enlève le label X redondant avec le titre
        
        # Gestion de la légende (uniquement sur le 1er graph pour alléger)
        if target_col and i > 0:
            if ax.get_legend():
                ax.get_legend().remove()
        elif target_col and i == 0:
             # On force l'affichage propre si seaborn l'a caché ou mal placé
             pass 

    # 3. Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    titre_global = f"Distributions Comparées (Target : {target_col})" if target_col else "Distributions Univariées"
    plt.suptitle(titre_global, y=1.02, fontsize=16)
    plt.show()

# --- Exemple ---
# vars_a_voir = ['size', 'population', 'revenu_median', 'nb_actes_delinquants']
# plot_kde_multivariables_grid(data_housing, vars_a_voir, target_col='target')
