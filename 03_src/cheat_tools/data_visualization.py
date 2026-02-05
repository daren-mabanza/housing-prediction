import seaborn as sns
import matplotlib.pyplot as plt
import math

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


import math

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



def plot_countplots_grid(df, columns=None, n_cols=3, figsize=(16, 4), rotate_xticks=45):
    """
    Affiche une grille de countplots pour les colonnes catégorielles.
    Idéal pour visualiser la répartition des modalités.


    Args:
        df (pd.DataFrame): Le dataframe.
        columns (list, optional): Liste des colonnes. Sinon, auto-détection.
        n_cols (int): Nombre de plots par ligne.
        figsize (tuple): Taille de base pour UNE ligne.
        rotate_xticks (int): Rotation des labels de l'axe x.
    """
    # 1. Auto-détection des catégorielles
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    n_vars = len(columns)
    if n_vars == 0:
        print("Aucune colonne catégorielle trouvée.")
        return

    # 2. Calcul de la grille
    n_rows = math.ceil(n_vars / n_cols)
    total_figsize = (figsize[0], figsize[1] * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=total_figsize,
        constrained_layout=True
    )

    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 3. Boucle de traçage
    for i, col in enumerate(columns):
        sns.countplot(
            data=df,
            x=col,
            ax=axes[i],
            order=df[col].value_counts().index,
            color='steelblue'
        )

        axes[i].set_title(f'Countplot : {col}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Effectif')
        axes[i].tick_params(axis='x', rotation=rotate_xticks)

    # 4. Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()

