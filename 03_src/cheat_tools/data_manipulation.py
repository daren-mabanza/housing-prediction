# ==================================
# Nom des fonctions
# ==================================
# multi_astype : Convertit le type de plusieurs colonnes d'un DataFrame
# multi_zfill : Formate des colonnes de codes en ajoutant des zeros devant
# imputation_modalite : Impute les valeurs manquantes de PLUSIEURS variables en utilisant une statistique par groupe.
# missing_percentage : Calcule le pourcentage de valeurs manquantes pour les variables spécifiées.
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
    
    print(f"Conversion de {len(columns)} colonne(s) en type '{dtype}'...\n")
    
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


