import great_expectations as gx
import pandas as pd

def initialisation_gx(context: gx.DataContext, dataframe: pd.DataFrame, ds_name: str, asset_name: str):
    """
    Initialise l'environnement Great Expectations V0.18.

    Args:
        context (gx.DataContext): Le contexte GX déjà initialisé.
        dataframe (pd.DataFrame): Le DataFrame Pandas contenant les données.
        ds_name (str): Nom de la Data Source.
        asset_name (str): Nom de l'Asset.

    Returns:
        batch_request: L'objet batch request pour créer le validator.
    """
    # 1. Data Source (V0.18 : utiliser add_or_update)
    # Cette méthode crée SI n'existe pas, sinon récupère l'existant
    datasource = context.sources.add_or_update_pandas(ds_name)
    print(f"✅ Datasource '{ds_name}' prête.")

    # 2. Asset (Rafraîchissement forcé)
    if asset_name in datasource.assets:
        datasource.assets.pop(asset_name)

    asset = datasource.add_dataframe_asset(asset_name, dataframe)
    print(f"📸 Asset '{asset_name}' mis à jour.")

    # 3. Batch Request
    batch_request = asset.build_batch_request()

    return batch_request


def afficher_resultats_validation(results):
    """
    Affiche le statut de chaque règle de validation Great Expectations.
    
    Args:
        results: L'objet retourné par validator.validate()
    """
    for i, result in enumerate(results.results, 1):
        status = "OK" if result.success else "KO"
        expectation_type = result.expectation_config.expectation_type
        print(f"Règle {i} ({expectation_type}): {status}")
