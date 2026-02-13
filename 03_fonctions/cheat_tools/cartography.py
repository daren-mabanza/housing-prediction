import plotly.express as px
import pandas as pd

def carte_interactive(df, hue, lat_col='approximate_latitude', lon_col='approximate_longitude', variable_info=None, titre=None, sample_size=20000):
    """
    Affiche une carte interactive avec la logique 'HUE' de Seaborn.
    
    Args:
        df (pd.DataFrame): Le dataframe.
        hue (str): La colonne qui détermine la COULEUR des points (ex: 'property_type').
        lat_col, lon_col (str): Colonnes de coordonnées.
        variable_info (str, optional): Variable supplémentaire à afficher au survol (ex: 'price').
        titre (str): Titre de la carte.
        sample_size (int): Limite de points pour la fluidité.
    """
    # 1. Gestion du titre
    if titre is None:
        titre = f"Carte répartie par : {hue}"

    # 2. Échantillonnage
    if len(df) > sample_size:
        print(f"⚠️ Dataset volumineux : affichage d'un échantillon de {sample_size} points.")
        df_plot = df.sample(sample_size, random_state=42).copy()
    else:
        df_plot = df.copy()

    # 3. Logique de Couleur (Discret vs Continu)
    # Pour 'hue', on veut souvent du discret (catégories bien séparées)
    is_numeric = pd.api.types.is_numeric_dtype(df_plot[hue])
    n_unique = df_plot[hue].nunique()
    
    # Si c'est un chiffre mais avec peu de valeurs (ex: Target 0/1, Nb Pièces 1..5), 
    # on force le mode Catégoriel pour avoir des couleurs distinctes et pas un dégradé.
    if is_numeric and n_unique < 15:
        df_plot[hue] = df_plot[hue].astype(str)
        print(f"Variable '{hue}' convertie en catégories pour l'affichage couleur.")
        color_scale = None
    elif is_numeric:
        color_scale = "Jet" # Dégradé pour prix/surface
    else:
        color_scale = None

    # 4. Préparation des données au survol (Tooltip)
    hover_dict = {lat_col: False, lon_col: False, hue: True}
    if variable_info and variable_info != hue:
        hover_dict[variable_info] = True

    # 5. Création de la Carte
    fig = px.scatter_mapbox(
        df_plot, 
        lat=lat_col, 
        lon=lon_col, 
        color=hue,                       # <--- C'est ici que 'hue' décide de la couleur
        color_continuous_scale=color_scale,
        size_max=10, 
        zoom=5,
        height=700,
        title=titre,
        opacity=0.7,
        hover_data=hover_dict            # <--- On affiche l'info supplémentaire ici
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    
    fig.show()

# --- CAS D'UTILISATION (Votre demande) ---

# CAS 1 : Target en fonction de Target (Carte colorée par Target 0/1)
# carte_interactive(data_housing, hue='target')

# CAS 2 : Target en fonction de Property Type (Carte colorée par Type de bien, mais on voit la Target au survol)
# carte_interactive(data_housing, hue='property_type', variable_info='target')

# CAS 3 : Prix en fonction de la Taille Agglo
# carte_interactive(data_housing, hue='taille_agglomeration', variable_info='price')
