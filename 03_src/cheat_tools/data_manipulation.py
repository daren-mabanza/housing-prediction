# ==================================
# Packages nécéssaires 
# ==================================
import pandas as pd
from typing import Union, List
# ==================================

# ==================================
# multi_astype
# ==================================

def multi_astype(df, columns, dtype, inplace=True):
    """
    Convertit le type de plusieurs colonnes d'un DataFrame
    
    Parameters
    ----------
    df : DataFrame
        Le DataFrame à modifier
    columns : list or str
        Liste des noms de colonnes à convertir (ou nom unique en str)
    dtype : str or type
        Type de données cible : 'str', 'int', 'float', 'category', 'datetime', etc.
    inplace : bool, default=True
        Si True, modifie le DataFrame original. Si False, retourne une copie
    
    Returns
    -------
    DataFrame or None
        DataFrame modifié (si inplace=False)
    
    Examples
    --------
    >>> convert_columns_type(data_housing, ['postal_code', 'city'], 'str')
    >>> convert_columns_type(data_housing, ['price', 'size'], 'float')
    >>> convert_columns_type(data_housing, 'property_type', 'category')
    """
    
    # Gère le cas d'une seule colonne (str au lieu de list)
    if isinstance(columns, str):
        columns = [columns]
    
    # DataFrame à modifier
    df_work = df if inplace else df.copy()
    
    # Vérifie que les colonnes existent
    missing_cols = [col for col in columns if col not in df_work.columns]
    if missing_cols:
        print(f"Colonnes inexistantes ignorées : {missing_cols}")
        columns = [col for col in columns if col in df_work.columns]
    
    if not columns:
        print("Aucune colonne valide à convertir")
        return None if not inplace else None
    
    # Mapping des types courants
    type_mapping = {
        'str': str,
        'string': str,
        'int': int,
        'integer': int,
        'float': float,
        'bool': bool,
        'boolean': bool,
        'category': 'category',
        'datetime': 'datetime64[ns]'
    }
    
    # Résout le type
    target_dtype = type_mapping.get(dtype, dtype)
    
    # Conversion avec gestion d'erreurs
    errors = []
    success = []
    
    for col in columns:
        try:
            # Type avant
            dtype_before = df_work[col].dtype
            
            # Conversion spéciale pour strings avec codes postaux
            if target_dtype == str:
                df_work[col] = df_work[col].astype('string')  # ← Avec guillemets
                # Si c'est un code postal, force 5 chiffres
                if 'postal' in col.lower() or 'cp' in col.lower() or 'code' in col.lower():
                    df_work[col] = df_work[col].str.replace('nan', '').str.zfill(5)
                    df_work[col] = df_work[col].replace('00000', None)
            
            # Conversion category
            elif target_dtype == 'category':
                df_work[col] = df_work[col].astype('category')
            
            # Conversion datetime
            elif target_dtype == 'datetime64[ns]':
                df_work[col] = pd.to_datetime(df_work[col], errors='coerce')
            
            # Autres conversions
            else:
                df_work[col] = df_work[col].astype(target_dtype)
            
            dtype_after = df_work[col].dtype
            success.append(col)
            print(f"{col}: {dtype_before} → {dtype_after}")
            
        except Exception as e:
            errors.append((col, str(e)))
            print(f"{col}: ERREUR - {str(e)}")
    
    # Résumé
    print(f"\n{'='*60}")
    print(f"Succès: {len(success)}/{len(columns)}")
    if errors:
        print(f"Échecs: {len(errors)}")
        for col, err in errors:
            print(f"   • {col}: {err}")
    
    return None if inplace else df_work


# ==================================
# multi_zfill
# ==================================


def multi_zfill(df, colonnes, n_digits):
    """
    Formate des colonnes de codes en ajoutant des zeros devant
    
    Parameters
    ----------
    df : DataFrame
        DataFrame a modifier
    colonnes : list ou str
        Nom(s) de(s) colonne(s) a formater
    n_digits : int, default=5
        Nombre de chiffres souhaite (ex: 5 pour codes INSEE)
    
    Returns
    -------
    DataFrame
        DataFrame modifie (inplace)
    
    Examples
    --------
    >>> formater_codes(data_housing, ['code_commune_INSEE', 'code_postal'], n_digits=5)
    >>> formater_codes(data_housing, 'code_departement', n_digits=2)
    """
    
    # Convertit en liste si une seule colonne
    if isinstance(colonnes, str):
        colonnes = [colonnes]
    
    # Applique zfill sur chaque colonne
    for col in colonnes:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(n_digits)
            print(f"Colonne '{col}' formatee sur {n_digits} chiffres")
        else:
            print(f"ATTENTION : Colonne '{col}' introuvable dans le DataFrame")


# ==================================
# imputation_modalite
# ==================================

def multi_imputation_modalite(data: pd.DataFrame,
                              group_by: str,
                              variables: Union[str, List[str]], # Accepte str ou liste
                              strategie: str = 'median') -> pd.DataFrame:
    """
    Impute les valeurs manquantes de PLUSIEURS variables en utilisant une statistique par groupe.
    """
    # 1. Copie de sécurité
    df = data.copy()
    
    # 2. Gestion flexible : si c'est une seule string, on la met dans une liste
    if isinstance(variables, str):
        variables = [variables]
    
    # 3. Vérification que la colonne de groupe existe
    if group_by not in df.columns:
        raise KeyError(f"La colonne de groupement '{group_by}' est introuvable.")

    # 4. Boucle sur chaque variable à imputer
    for var in variables:
        if var not in df.columns:
            # On peut choisir de lever une erreur ou juste un warning
            raise KeyError(f"Variable à imputer introuvable : {var}")
            
        # Le cœur du calcul (inchangé mais dans la boucle)
        imputed_values = df.groupby(group_by)[var].transform(strategie)
        df[var] = df[var].fillna(imputed_values)
    
    return df


# ==================================
# missing_percentage
# ==================================


def missing_percentage(data):

    df = data.copy()
    
    df = round(df.isna().sum()/len(df),2)

    df = pd.DataFrame(df).reset_index().rename(columns={"index":"features"})

    return df



# ===================================
# mapping_taille_agglomeration
# ===================================

def mapping_taille_agglomeration(df, colonne='taille_agglomeration'):
    """
    Simplifie les noms des modalités (version simple : mapping direct).
    """
    mapping_agglo = {
        "Commune appartenant à l'unité urbaine de Paris": "Paris (Agglo)",
        "Commune appartenant à une unité urbaine de 200 000 à 1 999 999 habitants": "200k - 2M hab",
        "Commune appartenant à une unité urbaine de 100 000 à 199 999 habitants": "100k - 200k hab",
        "Commune appartenant à une unité urbaine de 50 000 à 99 999 habitants": "50k - 100k hab",
        "Commune appartenant à une unité urbaine de 20 000 à 49 999 habitants": "20k - 50k hab",
        "Commune appartenant à une unité urbaine de 10 000 à 19 999 habitants": "10k - 20k hab",
        "Commune appartenant à une unité urbaine de 5 000 à 9 999 habitants": "5k - 10k hab",
        "Commune appartenant à une unité urbaine de 2 000 à 4 999 habitants": "2k - 5k hab",
        "Commune hors unité urbaine": "Rural / Hors Agglo"
    }

    # On convertit d'abord en string pour être sûr que le replace fonctionne
    # Puis on remplace les valeurs selon le dictionnaire
    df[colonne] = df[colonne].astype(str).replace(mapping_agglo)
    
    print(f"Transformation de '{colonne}' effectuée.")

# Utilisation
# mapping_taille_agglomeration(df)


# ===================================
# mapping_taille_pole_et_couronne
# ===================================

def mapping_taille_pole_et_couronne(df, colonne='taille_pole_et_couronne'):
    """
    Simplifie les noms des modalités (version simple : mapping direct).
    """
    mapping_dict = {
        "Aire de Paris": "Aire de Paris",
        "Aire de 700 000 habitants ou plus (hors Paris)": "Aire 700k+ (hors Paris)",
        "Aire de 200 000 à moins de 700 000 habitants": "Aire 200k - 700k",
        "Aire de 50 000 à moins de 200 000 habitants": "Aire 50k - 200k",
        "Aire de moins de 50 000 habitants": "Aire < 50k",
        "Commune hors attraction des villes": "Hors Attraction Villes"
    }

    # On convertit en string et on remplace ce qui correspond au dico
    # Ce qui n'est pas dans le dico reste inchangé
    df[colonne] = df[colonne].astype(str).replace(mapping_dict)

    print(f"Transformation de '{colonne}' effectuée.")


import pandas as pd

def fillna_multi(df, cols, value):
    """
    Remplit les NaN sur plusieurs colonnes avec une même valeur.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame à modifier (modifié en place).
    cols : str ou list of str
        Nom ou liste de noms de colonnes à remplir.
    value : any
        Valeur de remplacement (ex: "Missing", 0, "NA", etc.).

    Retour
    ------
    df : pd.DataFrame
        Le DataFrame (même objet) avec NaN remplacés sur les colonnes ciblées.
    """

    if isinstance(cols, str):
        cols = [cols]

    # Vérif basique
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"La colonne '{c}' n'existe pas dans le DataFrame.")

    df[cols] = df[cols].fillna(value)

    print("Remplissage des NaN effectué sur :")
    for c in cols:
        print(f"  - {c} -> '{value}'")


