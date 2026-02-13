# ==================================
# Nom des fonctions
# ==================================
# fusion_données_externes : Création d'un DataFrame avec des données de l'open data dans le cadre d'un projet de prédiction immobilière
# ==================================
# Packages nécéssaires 
# ==================================
from pathlib import Path
import pandas as pd
import duckdb
import great_expectations as gx
from cheat_tools.data_manipulation import multi_astype, multi_zfill, mapping_taille_agglomeration, mapping_taille_pole_et_couronne
from cheat_tools.geo_localisation import matching_coords_code_insee, calcul_proximite_services
from cheat_tools.data_quality import initialisation_gx, afficher_resultats_validation

ROOT = Path.cwd().parents[0]
RAW_DATA = ROOT / "01_data" / "01_raw"
PROCESSED_DATA = ROOT / "01_data" / "02_processed"

# ==================================

# ==================================
# fusion_donnees_externes
# ==================================

def fusion_donnees_externes():

    data_housing = pd.read_parquet(f"{RAW_DATA}/donnees_immobilieres.parquet")
    data = data_housing.copy()


    # --- Changement du type de certaines colonnes en "string"

    multi_astype(data,["property_type","city","exposition"],"string")
    
    # --- Matching coordonnées géographiques entre la table de base et "data_code_insee" pour récuperer le code INSEE
        # -- Import et récupération du code INSEE à partir de la table de l'INSEE 
    data_code_insee = (pd.read_parquet(RAW_DATA / "code_insee_24.parquet")
                       .astype({"code_commune_INSEE":"string",
                                "code_postal":"string",
                                "code_commune":"string",
                                "code_departement":"string",
                                "code_region":"string"}))
    

        # -- Ajout des 0 pour les variables de la table de l'INSEE (code en 01-09)
    multi_zfill(data_code_insee,["code_commune_INSEE","code_postal"],n_digits=5)

        # -- Association des données entre les deux tables grace aux coordonnées géographiques pour avoir le code insee dans "data_housing"
        # -- [FIN DE LA PREMIERE FUSION] 
    matching_coords_code_insee(
        df_housing=data,
        df_insee=data_code_insee,
        col_x_housing='approximate_longitude',      
        col_y_housing='approximate_latitude',       
        col_x_insee='longitude',                   
        col_y_insee='latitude',                     
        col_code_insee='code_commune_INSEE',        
        max_distance_km=50)

    print("Matching coordonnées - code insee : OK")
    print("="*50)

    # --- Préparation de la fusion avec les données de l'INSEE renseignant sur la population
        # -- Import des données de l'INSEE (population)
    data_population_insee = (pd.read_parquet(RAW_DATA / "population_24.parquet")
                             .astype({"codgeo":"string"}))
    
        # -- Recodage des codes insee (codgeo) et retrait des doublons potentiels
    multi_zfill(data_population_insee,["codgeo"],n_digits=5)
    data_population_insee = data_population_insee.drop_duplicates(subset=["codgeo"],keep="first")

        # -- Association des données entre les deux tables
        # -- [FIN DE LA DEUXIEME FUSION]
    data = pd.merge(data,
                    data_population_insee,
                    how = "left",
                    left_on = "code_insee",
                    right_on = "codgeo")
    
    print("Fusion avec 'data_population_insee' : OK")
    print("="*50)

    # --- Préparation de la fusion avec les données de l'INSEE renseignant sur le revenu médian par communes
        # -- Import des données de l'INSEE (revenu médian)
    data_revenu_insee = (pd.read_parquet(RAW_DATA / "revenu_median_24.parquet")
                         .astype({"Code gÃ©ographique":"string"}))                              
    

        # -- Recodage des codes insee (codgeo) et retrait des doublons potentiels
    multi_zfill(data_revenu_insee,["Code gÃ©ographique"],n_digits=5)
    data_revenu_insee = data_revenu_insee.drop_duplicates(subset=["Code gÃ©ographique"],keep="first")

    # -- Association des données entre les deux tables
    # [FIN DE LA TROISIEME FUSION]
    data = pd.merge(data,
                    data_revenu_insee,
                    how = "left",
                    left_on = "code_insee",
                    right_on = "Code gÃ©ographique")
    
    print("Fusion avec 'data_revenu_insee' : OK")
    print("="*50)


    # --- Préparation de la fusion avec les données de l'INSEE renseignant sur le niveau de délinquance par communes
        # -- Import des données de l'INSEE (délinquance)
    data_delinquance = (pd.read_parquet(RAW_DATA / "delinquance_24.parquet")
                        .astype({"CODGEO_2025":"string"}))

        # -- Retrait des doublons potentiels
    data_delinquance = data_delinquance.drop_duplicates(subset=["CODGEO_2025"],keep="first")

    # Association des données entre les deux tables 
        # -- [FIN DE LA QUATRIEME FUSION]
    data = pd.merge(data,
                    data_delinquance,
                    how = "left",
                    left_on = "code_insee",
                    right_on = "CODGEO_2025")


    print("Fusion avec 'data_delinquance' : OK")
    print("="*50)


    # --- Préparation de la fusion avec les données de l'INSEE renseignant sur le statut urbain par communes
        # -- Import des données de l'INSEE (statu urbain)
    data_urban = (pd.read_parquet(RAW_DATA / "info_urbaines_24.parquet").
                  astype({"CODGEO":"string"}))

        # -- Retrait des doublons potentiels
    data_urban = data_urban.drop_duplicates(subset=["CODGEO"],keep="first")

        # -- Association des données entre les deux tables
        # -- [FIN DE CINQUIEME FUSION]
    data = pd.merge(data,
                    data_urban,
                    how = "left",
                    left_on = "code_insee",
                    right_on = "CODGEO")
    
    print("Fusion avec 'data_urban' : OK")
    print("="*50)

    
    # --- Préparation de la fusion avec les données de l'INSEE renseignant sur le statut touristique par communes
        # -- Import des données de l'INSEE (statut touristique)
    data_statut_tourisme = (pd.read_parquet(RAW_DATA / "statut_touristique_24.parquet")
                            .astype({"code_geo":"string"}))

        # -- Retrait des doublons potentiels
    data_statut_tourisme = data_statut_tourisme.drop_duplicates(subset=["code_geo"],keep="first")

        # -- Association des données entre les deux tables
        # -- [FIN DE LA SIXIEME FUSION]
    data = pd.merge(data,
                    data_statut_tourisme,
                    how = "left",
                    left_on = "code_insee",
                    right_on = "code_geo")
    
    print("Fusion avec 'data_statut_tourisme' : OK")
    print("="*50)


    # --- Préparation de la fusion avec les données de l'INSEE renseignant sur ls infrastructures touristiques par communes
        # -- Import des données de l'INSEE (infrastrcutures touristiques)
    data_infra_tourisme = (pd.read_parquet(RAW_DATA / "infrastructures_tourisme_24.parquet")
                           .astype({"code_insee_com":"string"}))

        # -- Retrait des doublons potentiels
    data_infra_tourisme = data_infra_tourisme.drop_duplicates(subset=["code_insee_com"],keep="first")

        # -- Association des données entre les deux tables
        # -- [FIN DE LA SEPTIEME FUSION]
    data = pd.merge(data,
                    data_infra_tourisme,
                    how = "left",
                    left_on = "code_insee",
                    right_on = "code_insee_com")
    
    print("Fusion avec 'data_infra_tourisme' : OK")
    print("="*50)


    # --- Récupération des commodités principales les plus proches pour chaque logement de la base de donnée "data"
        # -- Chargment de la table "Base permanente des équipements 2024" 

    con = duckdb.connect()

    data_bpe = con.sql(f"""
        SELECT AN, NOMRS, 
               CODPOS::VARCHAR CODPOS, 
               DEPCOM::VARCHAR DEPCOM, 
               DEP::VARCHAR DEP, 
               REG::VARCHAR REG,
               LIBCOM, TYPEQU, LONGITUDE, LATITUDE
                   
        FROM read_parquet('{ROOT}/01_data/01_raw/bpe_24.parquet')
                   
        WHERE TYPEQU IN ('E107', 'E108', 'E109',
                        'B201', 'B202', 'B207', 'B104', 'B105',
                        'C107', 'C108', 'C109', 'C201', 'C301', 'C302',
                        'D265', 'D108', 'D113', 'D307',
                        'F303', 'F307', 'B307')
                    AND DEP not in ('971','972','973','974','976')
        """).df()
    

    # --- Récuperer la distance la plus courte aux commodités pouvant influencer le prix d'un bien immo
    calcul_proximite_services(
        data,
        data_bpe,
        col_lat_principal='approximate_latitude',
        col_lon_principal='approximate_longitude',
        col_lat_equipement='LATITUDE',
        col_lon_equipement='LONGITUDE',
        col_typequ='TYPEQU',
        categories_typequ=None,
        rayon_densite_km=1.5
    )

    print("Calcul de la distance entre les logements et les commodités les plus proches : OK")
    print("="*50)

    # --- Renommage de certaines modalités 
    # [TYPE_COMMUNE_BV2022]
    data["TYPE_COMMUNE_BV2022"] = (data["TYPE_COMMUNE_BV2022"]
                                   .replace({"Pôle partiel":"Non pôle",
                                             "Commune associée à un pôle partiel":"Non pôle"}))

    # [Type Touristique]
    data["Type Touristique"] = data["Type Touristique"].fillna("Non classé touristique")

    multi_zfill(data,"id_annonce",n_digits=5)


    # --- Renommage de certaines variables 
    data = data.rename(columns={
                                 "p21_pop":"population",
                                 "[DISP] Nbre de mÃ©nages fiscaux":"nb_menages_fiscaux",
                                 "[DISP] MÃ©diane (â‚¬)":"revenu_median",
                                 "nombre":"nb_actes_delinquants",
                                 "Typo.degré.densité":"type_degre_densite",
                                 "Typo.rural.urbain":"type_rural_urbain",
                                 "Type Touristique":"type_touristique",
                                 "sup350k":"target",
                                 "STATUT_COM_UU":"position_commune_unite_urbaine",
                                 "TUU2017":"taille_agglomeration",
                                 "TYPE_UU2020":"type_unite_urbaine",
                                 "TAAV2017":"taille_pole_et_couronne",
                                 "TYPE_COMMUNE_BV2022":"role_commune_bassin_de_vie"})


    # --- Préselection des variables pour le modèle
    variables = ["id_annonce","code_insee","property_type","approximate_latitude","approximate_longitude","size","floor",
                 "land_size","energy_performance_value","ghg_value","nb_rooms",
                 "nb_bedrooms","nb_bathrooms","nb_parking_places",
                 "nb_boxes","nb_photos","has_a_balcony","nb_terraces",
                 "has_a_cellar","has_a_garage","has_air_conditioning",
                 "last_floor","upper_floors","population","nb_menages_fiscaux",
                 "revenu_median","nb_actes_delinquants","position_commune_unite_urbaine",
                 "taille_agglomeration","type_unite_urbaine","taille_pole_et_couronne",
                 "role_commune_bassin_de_vie","type_degre_densite","type_rural_urbain",
                 "type_touristique","nb_hebergement_tourisme","nb_lits_tourisme","nb_village_vacances",
                 "nb_lits_village_vacances","nb_residence_tourisme","nb_lits_residence_tourisme",
                 "proximite_gare","proximite_commerce_alimentaire","proximite_education_primaire",
                 "proximite_education_secondaire","proximite_sante","densite_services_rayon","target"]
    
    data = data[variables]

    mapping_taille_pole_et_couronne(data)
    mapping_taille_agglomeration(data)

    multi_astype(data,["code_insee","id_annonce"],"string")
    multi_astype(data, list(data.select_dtypes("object").columns) + ["property_type"], "category")

    print("Renommages et modifications des types : OK")
    print("="*50)

    # Check de la qualité des données (Great Expectation)
    context = gx.get_context()

    batch_request = initialisation_gx(
        context=context,
        dataframe=data,
        ds_name="housing_project",
        asset_name="housing_open_data"
        )

    validator = context.get_validator(batch_request=batch_request)

        # Création des règles
    validator.expect_column_values_to_not_be_null('target')  
    validator.expect_compound_columns_to_be_unique(['approximate_latitude', 'approximate_longitude'])
    validator.expect_column_value_lengths_to_be_between('code_insee', 5, 5)
    validator.expect_column_value_lengths_to_be_between('id_annonce', 5, 5)

        # Validation des résultats et export de la table sous condition de validité des données
    results = validator.validate()

    afficher_resultats_validation(results)


    if results.success == True:
        data.to_csv(PROCESSED_DATA / "housing_open_data_processed.csv", index = False)
        print("Check de la validité des données : OK")
        print("Export de la table : OK")

    else:
        print("Données non valides !")


    return data