# Projet de Classification de Biens Immobiliers : Workflow Data Science Complet

## Présentation du projet

Ce projet a pour objectif de prédire si un bien immobilier appartient à la catégorie des biens à prix élevé à partir :

- des caractéristiques intrinsèques du bien (surface, nombre de pièces, type de bien, etc.),
- du contexte socio-économique local (revenu médian, population, criminalité),
- d’indicateurs d’attractivité territoriale (tourisme, densité de services),
- et de données open data issues de data.gouv (INSEE, statistiques locales, équipements publics).

L’ambition n’est pas uniquement prédictive. Le projet est conçu comme un **workflow data science structuré**, allant de la donnée brute à un modèle interprétable et industrialisable.

---

## Organisation du projet

```
01_data/          → Données brutes et données transformées
02_notebooks/     → Exploration, économétrie, machine learning, SHAP
03_fonctions/     → Modules Python (merge, cleaning, feature engineering, modèle, script complet)
04_model/         → Modèle entraîné sauvegardé
```

La logique analytique (EDA, économétrie, interprétation) est séparée de la logique métier réutilisable (fonctions Python), afin de garantir :

- modularité,
- traçabilité,
- reproductibilité,
- possibilité d’exécution automatisée.

---

## Workflow Data Science

### 1. Intégration et enrichissement des données

Les annonces immobilières sont enrichies via des jeux de données open data (data.gouv), notamment :

- population et revenu médian (INSEE),
- indicateurs de délinquance,
- données touristiques,
- densité de services (Base Permanente des Équipements),
- typologies territoriales (rural/urbain, taille d’agglomération).

Cette étape permet de passer d’un dataset transactionnel à un dataset contextualisé territorialement.

Un contrôle qualité systématique est appliqué via Great Expectations pour sécuriser la structure et la cohérence des données.

---

### 2. Analyse exploratoire et nettoyage

L’EDA permet :

- d’identifier les distributions,
- de détecter les valeurs aberrantes,
- d’analyser les relations avec la variable cible.

Le nettoyage inclut :

- correction des valeurs incohérentes (surface, nombre de pièces),
- harmonisation des modalités catégorielles,
- réduction aux variables pertinentes,
- validations structurelles (unicité, non-nullité, cohérence des formats).

Chaque étape est validée avant de passer à la suivante.

---

### 3. Analyse économétrique

Un modèle logit est estimé afin de comprendre les déterminants du prix élevé.

Cette phase permet :

- d’identifier les variables statistiquement significatives,
- d’interpréter les effets marginaux,
- de détecter des non-linéarités.

Les résultats mettent en évidence le rôle structurant :

- du nombre de pièces,
- de la surface,
- du revenu médian local,
- de la localisation (notamment Paris),
- de certains indicateurs territoriaux.

L’économétrie sert ici de socle explicatif et guide le feature engineering.

---

### 4. Feature Engineering

À partir des enseignements précédents :

- centrage des variables numériques,
- ajout de termes quadratiques,
- binarisation du nombre de pièces,
- transformation logarithmique du revenu médian,
- construction d’indicateurs synthétiques (criminalité, attractivité touristique).

Un dataset final cohérent est ainsi constitué pour la modélisation prédictive.

---

### 5. Machine Learning

Une pipeline scikit-learn complète est implémentée :

- imputation via `IterativeImputer` (estimateur XGBoost),
- standardisation des variables numériques,
- encodage One-Hot des variables catégorielles,
- régression logistique pénalisée L2,
- optimisation des hyperparamètres via `RandomizedSearchCV`.

#### Performances

- ROC-AUC ≈ 0.90  
- Average Precision ≈ 0.85  
- Brier Score ≈ 0.117  

Le modèle présente une excellente capacité de discrimination, une bonne calibration et une stabilité satisfaisante entre train et test.

---

### 6. Interprétabilité

L’analyse SHAP permet :

- d’identifier les variables les plus influentes globalement,
- d’analyser le sens des contributions,
- d’expliquer individuellement les prédictions.

Les résultats confirment la cohérence entre approche économétrique et modèle prédictif.

---

## Exécution

La pipeline complète peut être exécutée via le script présent dans `03_fonctions/script_complet.py` ou directement depuis le notebook `09_script_complet.ipynb`.

Le modèle final est sauvegardé dans :

```
04_model/housing_logit_model.joblib
```

---

## Remarque importante

Le projet n’est pas entièrement clonable en l’état car le fichier suivant est exclu du repository (volume trop important) :

```
01_data/01_raw/bpe_24.parquet
```

Ce fichier correspond aux données de la Base Permanente des Équipements (BPE) et doit être téléchargé séparément depuis data.gouv pour exécuter la pipeline complète.

---

## Conclusion

Ce projet illustre un workflow data science complet et cohérent :

- intégration de données open data,
- contrôle qualité rigoureux,
- analyse explicative,
- optimisation prédictive,
- interprétabilité avancée,
- automatisation end-to-end.

Il démontre la capacité à structurer un projet depuis la donnée brute jusqu’à un modèle performant, explicable et prêt à être utilisé dans un cadre décisionnel.
