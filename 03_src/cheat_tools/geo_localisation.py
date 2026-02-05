# ==================================
# Nom des fonctions
# ==================================
# matching_coords_code_insee : Associe code INSEE en fonction de la ville la plus proche
# calcul_proximite_services : Calcule les distances minimales à différents types de services et la densité de services à proximité.
# ==================================
# Packages nécéssaires 
# ==================================
import pandas as pd
import numpy as np

from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree


# ==================================


# ====================================
# matching_coords_code_insee
# ====================================

def matching_coords_code_insee(df_housing, 
                               df_insee, 
                               col_x_housing, 
                               col_y_housing,
                               col_x_insee, 
                               col_y_insee,
                               col_code_insee,
                               max_distance_km=50):
    """
    Associe code INSEE en fonction de la ville la plus proche
    
    Parameters
    ----------
    df_housing : DataFrame
        Table logements avec coordonnees
    df_insee : DataFrame
        Table INSEE avec coordonnees et codes communes
    col_x_housing : str
        Nom colonne longitude/x dans df_housing
    col_y_housing : str
        Nom colonne latitude/y dans df_housing
    col_x_insee : str
        Nom colonne longitude/x dans df_insee
    col_y_insee : str
        Nom colonne latitude/y dans df_insee
    col_code_insee : str
        Nom colonne code commune INSEE dans df_insee
    max_distance_km : float, default=50
        Distance max acceptee (en km)
    
    Returns
    -------
    DataFrame
        df_housing enrichi avec code_insee et distance_ville_m
    """

    
    
    print(f"Association codes INSEE par proximite...")
    print(f"Logements : {len(df_housing)}")
    print(f"Villes INSEE : {len(df_insee)}")
    
    # Conversion GeoDataFrame
    geom_housing = [Point(xy) for xy in zip(df_housing[col_x_housing], df_housing[col_y_housing])]
    gdf_housing = gpd.GeoDataFrame(df_housing, geometry=geom_housing, crs="EPSG:4326")
    
    geom_insee = [Point(xy) for xy in zip(df_insee[col_x_insee], df_insee[col_y_insee])]
    gdf_insee = gpd.GeoDataFrame(df_insee, geometry=geom_insee, crs="EPSG:4326")
    
    # Jointure nearest
    result = gpd.sjoin_nearest(
        gdf_housing,
        gdf_insee[[col_code_insee, 'geometry']],
        how='left',
        max_distance=max_distance_km * 1000,
        distance_col='distance_m'
    )
    
    # Gestion duplicatas
    if result.index.duplicated().sum() > 0:
        result = result[~result.index.duplicated(keep='first')]
    
    # Recuperation
    df_housing['code_insee'] = result[col_code_insee].values
    df_housing['distance_ville_m'] = result['distance_m'].values
    
    # Stats
    found = df_housing['code_insee'].notna().sum()
    avg_dist = df_housing['distance_ville_m'].mean()
    max_dist = df_housing['distance_ville_m'].max()
    
    print(f"  - Appariements : {found}/{len(df_housing)} ({found/len(df_housing)*100:.1f}%)")

# Utilisation - TU CHOISIS TES NOMS DE COLONNES
#associer_code_insee_nearest(
#    df_housing=data_housing,
#    df_insee=table_insee,
#    col_x_housing='approximate_longitude',      # TON NOM
#    col_y_housing='approximate_latitude',       # TON NOM
#    col_x_insee='longitude',                    # TON NOM
#    col_y_insee='latitude',                     # TON NOM
#    col_code_insee='code_commune_insee',        # TON NOM
#    max_distance_km=50
#)



# ====================================
# calcul_proximite_services
# ====================================

def calcul_proximite_services(
    df_principal,
    df_equipements,
    col_lat_principal='latitude',
    col_lon_principal='longitude',
    col_lat_equipement='latitude',
    col_lon_equipement='longitude',
    col_typequ='TYPEQU',
    categories_typequ=None,
    rayon_densite_km=1.5
):
    """
    Calcule les distances minimales à différents types de services et la densité de services à proximité.
    Modifie df_principal en place en ajoutant les nouvelles colonnes.
    Optimisé pour de gros volumes (280k équipements, 30k logements).
    
    Paramètres:
    -----------
    df_principal : pd.DataFrame
        DataFrame principal où ajouter les nouvelles variables (logements) - modifié en place
    df_equipements : pd.DataFrame
        DataFrame contenant les équipements/services avec leurs coordonnées
    col_lat_principal : str
        Nom de la colonne latitude dans df_principal
    col_lon_principal : str
        Nom de la colonne longitude dans df_principal
    col_lat_equipement : str
        Nom de la colonne latitude dans df_equipements
    col_lon_equipement : str
        Nom de la colonne longitude dans df_equipements
    col_typequ : str
        Nom de la colonne contenant les types d'équipements
    categories_typequ : dict
        Dictionnaire {nom_variable: [liste_codes_TYPEQU]}
        Exemple: {'proximite_gare': ['E107', 'E108', 'E109'], ...}
    rayon_densite_km : float
        Rayon en km pour calculer la densité de services
    
    Retourne:
    ---------
    None : Modifie df_principal en place
    """
    
    if categories_typequ is None:
        categories_typequ = {
            'proximite_gare': ['E107', 'E108', 'E109'],
            'proximite_commerce_alimentaire': ['B201', 'B202', 'B207', 'B104', 'B105'],
            'proximite_education_primaire': ['C107', 'C108', 'C109', 'C201', 'C301', 'C302'],
            'proximite_education_secondaire': ['D265', 'D108', 'D113', 'D307'],
            'proximite_sante': ['F303', 'F307', 'B307']
        }
    
    print(f"Traitement de {len(df_principal)} logements et {len(df_equipements)} equipements...")
    
    # Filtrer les coordonnées invalides (NaN, inf) dans df_principal
    mask_valid_principal = (
        df_principal[col_lat_principal].notna() & 
        df_principal[col_lon_principal].notna() &
        np.isfinite(df_principal[col_lat_principal]) &
        np.isfinite(df_principal[col_lon_principal])
    )
    
    nb_invalid_principal = (~mask_valid_principal).sum()
    if nb_invalid_principal > 0:
        print(f"Attention: {nb_invalid_principal} logements avec coordonnees invalides (seront mis a NaN)")
    
    # Filtrer les coordonnées invalides dans df_equipements
    mask_valid_equipements = (
        df_equipements[col_lat_equipement].notna() & 
        df_equipements[col_lon_equipement].notna() &
        np.isfinite(df_equipements[col_lat_equipement]) &
        np.isfinite(df_equipements[col_lon_equipement])
    )
    
    nb_invalid_equipements = (~mask_valid_equipements).sum()
    if nb_invalid_equipements > 0:
        print(f"Attention: {nb_invalid_equipements} equipements avec coordonnees invalides (ignores)")
    
    df_equipements_clean = df_equipements[mask_valid_equipements].copy()
    
    print(f"Donnees valides : {mask_valid_principal.sum()} logements, {len(df_equipements_clean)} equipements")
    
    # Initialiser les colonnes avec NaN
    for nom_variable in list(categories_typequ.keys()) + ['densite_services_rayon']:
        df_principal[nom_variable] = np.nan
    
    # Si aucune donnée valide, arrêter ici
    if mask_valid_principal.sum() == 0 or len(df_equipements_clean) == 0:
        print("Pas de donnees valides a traiter")
        return
    
    print("Creation des GeoDataFrames et conversion en Lambert 93...")
    
    # Créer GeoDataFrame uniquement avec les données valides
    df_principal_valid = df_principal[mask_valid_principal].copy()
    
    gdf_principal = gpd.GeoDataFrame(
        df_principal_valid,
        geometry=gpd.points_from_xy(df_principal_valid[col_lon_principal], df_principal_valid[col_lat_principal]),
        crs="EPSG:4326"
    )
    gdf_principal = gdf_principal.to_crs("EPSG:2154")
    
    gdf_equipements = gpd.GeoDataFrame(
        df_equipements_clean,
        geometry=gpd.points_from_xy(df_equipements_clean[col_lon_equipement], df_equipements_clean[col_lat_equipement]),
        crs="EPSG:4326"
    )
    gdf_equipements = gdf_equipements.to_crs("EPSG:2154")
    
    coords_principal = np.array([[point.x, point.y] for point in gdf_principal.geometry])
    coords_equipements = np.array([[point.x, point.y] for point in gdf_equipements.geometry])
    
    for nom_variable, codes_typequ in categories_typequ.items():
        print(f"Calcul de {nom_variable}...")
        
        mask_type = df_equipements_clean[col_typequ].isin(codes_typequ)
        coords_filtre = coords_equipements[mask_type]
        
        if len(coords_filtre) == 0:
            print(f"  Attention: Aucun equipement trouve pour {nom_variable}")
            continue
        
        print(f"  -> {len(coords_filtre)} equipements trouves pour cette categorie")
        
        tree = cKDTree(coords_filtre)
        distances, indices = tree.query(coords_principal, k=1)
        
        # Assigner les distances uniquement aux lignes valides
        df_principal.loc[mask_valid_principal, nom_variable] = distances
    
    print(f"\nCalcul de la densite de services dans un rayon de {rayon_densite_km} km...")
    
    tree_all = cKDTree(coords_equipements)
    rayon_metres = rayon_densite_km * 1000
    counts = tree_all.query_ball_point(coords_principal, r=rayon_metres, return_length=True)
    
    # Assigner les counts uniquement aux lignes valides
    df_principal.loc[mask_valid_principal, 'densite_services_rayon'] = counts
    
    print("\nTermine !")
    print(f"\nColonnes creees:")
    for col in list(categories_typequ.keys()) + ['densite_services_rayon']:
        print(f"  - {col}")
