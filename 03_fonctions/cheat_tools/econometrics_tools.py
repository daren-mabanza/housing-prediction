import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def box_tidwell_test(df, y, continuous_vars, add_other_predictors=None, alpha=0.05, eps=1e-6):
    """
    Box-Tidwell pour tester la linéarité du logit pour des variables continues.
    Gère les valeurs à 0 en ajoutant un petit décalage eps.
    """

    df_bt = df.copy()

    # 1) Vérification valeurs négatives uniquement
    for v in continuous_vars:
        if (df_bt[v] < 0).any():
            raise ValueError(
                f"La variable '{v}' contient des valeurs < 0 ; "
                "corrige/la avant le test (Box-Tidwell ne supporte pas les valeurs négatives)."
            )

    # 2) Création des termes X * ln(X) avec décalage si nécessaire
    interaction_terms = []
    for v in continuous_vars:
        x = df_bt[v].astype(float)
        # Décalage si min <= 0 (0 -> eps, etc.)
        if (x <= 0).any():
            x_shifted = x + eps
        else:
            x_shifted = x
        new_name = f"{v}_ln"
        df_bt[new_name] = x_shifted * np.log(x_shifted)
        interaction_terms.append(new_name)

    # 3) Construction de la formule
    rhs_terms = list(continuous_vars) + interaction_terms
    if add_other_predictors is not None:
        rhs_terms += list(add_other_predictors)

    formula_bt = y + " ~ " + " + ".join(rhs_terms)

    # 4) Estimation du logit avec termes Box-Tidwell
    model_bt = smf.logit(formula_bt, data=df_bt).fit(disp=False)

    # 5) Récupération des stats pour chaque terme X*ln(X)
    summary_frame = model_bt.summary2().tables[1]
    bt_rows = summary_frame.loc[interaction_terms][["Coef.", "Std.Err.", "z", "P>|z|"]]
    bt_rows = bt_rows.rename(columns={"P>|z|": "p_value"})
    bt_rows.index = [name.replace("_ln", "") for name in bt_rows.index]

    # 6) Recommandation de transformation
    bt_rows["transformation_necessaire"] = bt_rows["p_value"] < alpha

    return bt_rows






from scipy import stats
import pandas as pd

def trv_logit(model_full, model_restricted, model_names=None, verbose=True):
    """
    Likelihood Ratio Test (LRT) entre deux modèles logit imbriqués.

    Paramètres
    ----------
    model_full : statsmodels LogitResults
        Modèle complet (plus de paramètres).
    model_restricted : statsmodels LogitResults
        Modèle restreint (imbriqué dans model_full).
    model_names : tuple of str or None
        Noms des modèles pour affichage (ex: ('Full', 'Restreint')).
        Si None, utilise 'Full model' et 'Restricted model'.
    verbose : bool
        Si True, affiche les résultats.

    Retour
    ------
    results : dict
        {'lr_stat': float, 'df_diff': int, 'p_value': float,
         'll_full': float, 'll_restricted': float}
    """
    if model_names is None:
        model_names = ('Full model', 'Restricted model')

    ll_full = model_full.llf
    ll_restr = model_restricted.llf
    lr_stat = 2 * (ll_full - ll_restr)
    df_diff = model_full.df_model - model_restricted.df_model
    p_lrt = stats.chi2.sf(lr_stat, df_diff)

    results = {
        'lr_stat': lr_stat,
        'df_diff': df_diff,
        'p_value': p_lrt,
        'll_full': ll_full,
        'll_restricted': ll_restr
    }

    if verbose:
        print(f"\n=== Likelihood Ratio Test ===")
        print(f"{model_names[0]} vs {model_names[1]}")
        print(f"LogLik {model_names[0]} : {ll_full:.4f}")
        print(f"LogLik {model_names[1]} : {ll_restr:.4f}")
        print(f"Chi² LR           : {lr_stat:.4f}")
        print(f"df difference     : {df_diff}")
        print(f"p-value           : {p_lrt:.4g}")
        print("Interprétation : p < 0.05 → modèle complet significativement meilleur.")

    return results


# ---------- Exemple 1 : ton cas classique ----------
#logit_restricted = smf.logit("y ~ x1 + x2", data=df).fit()
#lrt_res1 = lrt_logit(
#    model_full=logit,
#    model_restricted=logit_restricted,
#    model_names=("Complet (x1+x2+x3)", "Restreint (x1+x2)")
#)

# ---------- Exemple 2 : test d'un seul paramètre ----------
#logit_no_x3 = smf.logit("y ~ x1 + x2", data=df).fit()
#lrt_res2 = lrt_logit(logit, logit_no_x3, model_names=("Avec x3", "Sans x3"))

# ---------- Exemple 3 : test de plusieurs paramètres ----------
#logit_base = smf.logit("y ~ x1", data=df).fit()
#lrt_res3 = lrt_logit(logit, logit_base, model_names=("Complet", "Seulement x1"))

# Récup des résultats pour tableaux/rapports
#print("\nRésultats LRT x3 seul :", lrt_res2['p_value'])





import pandas as pd
import numpy as np
from scipy import stats

def hosmer_lemeshow_test(model, df=None, y=None, g=10, verbose=True):
    """
    Test de Hosmer-Lemeshow pour la calibration d'un modèle logit.

    Paramètres
    ----------
    model : statsmodels LogitResults
        Modèle logit déjà estimé.
    df, y : str or None
        Si model n'a pas les données, fournir df et nom de y pour calculer les résidus.
        Sinon, utilise model.resid_response.
    g : int
        Nombre de groupes (déciles par défaut = 10).
    verbose : bool
        Si True, affiche tableau observé/attendu et résultats du test.

    Retour
    ------
    results : dict
        {'hl_stat': float, 'df': int, 'p_value': float, 'table_oe': pd.DataFrame}
    """
    if df is not None and y is not None:
        y_true = df[y].values
        y_prob = model.predict(df)
    else:
        y_true = model.model.endog.flatten()
        y_prob = model.fittedvalues

    # Création des groupes via quantiles des prédictions
    hl_data = pd.DataFrame({"y": y_true, "p": y_prob})
    hl_data["group"] = pd.qcut(hl_data["p"], q=g, labels=range(1, g+1), duplicates="drop")

    # Agrégats par groupe : observé, attendu, effectif
    oe_summary = hl_data.groupby("group").agg(
        n_total=("y", "count"),
        y_obs=("y", "sum"),
        p_exp=("p", "sum")
    ).reset_index()
    oe_summary["y_exp"] = oe_summary["p_exp"]
    oe_summary["residus"] = oe_summary["y_obs"] - oe_summary["y_exp"]

    # Statistique Hosmer-Lemeshow
    hl_stat = ((oe_summary["residus"] ** 2) / 
               (oe_summary["y_exp"] * (oe_summary["n_total"] - oe_summary["y_exp"]))).sum()
    df_hl = g - 2
    p_hl = stats.chi2.sf(hl_stat, df_hl)

    results = {
        'hl_stat': hl_stat,
        'df': df_hl,
        'p_value': p_hl,
        'table_oe': oe_summary.round(3)
    }

    if verbose:
        print(f"\n=== Test de Hosmer-Lemeshow (g={g} groupes) ===")
        print(results['table_oe'].to_string(index=False))
        print(f"\nHL stat  : {hl_stat:.4f}")
        print(f"df       : {df_hl}")
        print(f"p-value  : {p_hl:.4g}")
        print("Interprétation : p > 0.05 → bonne calibration (H0 non rejetée).")

    return results



# ---------- Test HL sur ton modèle ----------
#hl_res = hosmer_lemeshow_test(
#    model=logit,
#    g=10,  # déciles
#    verbose=True
#)

# ---------- Variante : test avec plus/moins de groupes ----------
#hl_8 = hosmer_lemeshow_test(logit, g=8)
#hl_5 = hosmer_lemeshow_test(logit, g=5)

# ---------- Accès au tableau observé/attendu pour rapports ----------
#print("\nTableau O/E :", hl_res['table_oe'])
#print("p-value HL :", hl_res['p_value'])


from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def vif_logit(model, df=None, formula=None, drop_const=True):
    """
    Variance Inflation Factor (VIF) pour détecter la multicolinéarité.
    Retourne uniquement un DataFrame avec les VIF.
    """

    # 1) Essayer d'utiliser directement la matrice X du modèle
    exog = getattr(model.model, "exog", None)
    exog_names = getattr(model.model, "exog_names", None)

    if exog is not None and exog_names is not None and df is None:
        X_df = pd.DataFrame(exog, columns=exog_names)
    else:
        # 2) Sinon, reconstruire X à partir de la formule + df
        if formula is None:
            if hasattr(model, "model") and hasattr(model.model, "formula"):
                formula = model.model.formula
            else:
                raise ValueError(
                    "Fournir 'formula' ou un model avec .model.formula, "
                    "ou appeler vif_logit(model, df=None) pour utiliser model.model.exog"
                )

        if df is None:
            raise ValueError("Vous devez fournir 'df' si vous voulez reconstruire X avec 'formula'.")

        try:
            _, X_df = dmatrices(formula, data=df, return_type='dataframe')
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors de la construction de X via dmatrices. "
                f"Vérifiez la formule et les variables dans df.\nDétail : {e}"
            )

    # 3) Suppression de la constante si demandée
    for const_name in ["Intercept", "const"]:
        if drop_const and const_name in X_df.columns:
            X_df = X_df.drop(columns=const_name)

    # 4) Calcul des VIF
    vif_data = pd.DataFrame({
        "variable": X_df.columns,
        "VIF": [variance_inflation_factor(X_df.values, i)
                for i in range(X_df.shape[1])]
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    return vif_data







# ---------- Exemple 1 : sur ton modèle existant ----------
#vif_res = vif_logit(
#    model=logit,  # ton objet logit déjà estimé
#    verbose=True
#)

# ---------- Exemple 2 : avec formule explicite (modèle + complexe) ----------
#vif_complex = vif_logit(
#    model=None,  # pas besoin du modèle
#    df=df,
#    formula="y ~ x1 + x2 + x3 + np.log(x1):x2",  # avec interaction
#    verbose=True
#)

# ---------- Exemple 3 : test sur modèle restreint ----------
#logit_restr = smf.logit("y ~ x1 + x2", data=df).fit()
#vif_restr = vif_logit(logit_restr)

# ---------- Analyse automatisée ----------
#print("Variables les plus corrélées :", vif_res.nlargest(3, 'VIF')['variable'].tolist())
#print("VIF max :", vif_res['VIF'].max())




import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt

def spline_test_multi(
    df,
    formula_base,
    vars_list,
    df_spline=4,
    at="mean",
    plot=True,
    alpha=0.05
):
    """
    Test spline pour plusieurs variables continues dans un modèle logit.
    Trace les courbes spline vs linéaire dans une grille (3 plots par ligne).
    """

    results = {}

    for var in vars_list:
        if var not in formula_base:
            raise ValueError(f"La variable '{var}' doit apparaître dans 'formula_base'.")

        formula_lin = formula_base
        replace_patterns = [
            f"+ {var} +",
            f"+ {var}",
            f"{var} +",
            f"~ {var} +",
            f"~ {var}"
        ]

        formula_spline = formula_base
        for pat in replace_patterns:
            if pat in formula_spline:
                formula_spline = formula_spline.replace(
                    pat,
                    pat.replace(var, f"bs({var}, df={df_spline})")
                )

        if var not in formula_lin:
            raise ValueError(f"Impossible de reconstruire une formule linéaire propre pour '{var}'.")

        if f"bs({var}, df={df_spline})" not in formula_spline:
            raise ValueError(
                f"La formule spline n'a pas été correctement construite pour '{var}'. "
                f"Vérifie 'formula_base' et le nom de la variable."
            )

        model_lin = smf.logit(formula_lin, data=df).fit(disp=False)
        model_spline = smf.logit(formula_spline, data=df).fit(disp=False)

        ll_lin = model_lin.llf
        ll_spl = model_spline.llf
        lr_stat = 2 * (ll_spl - ll_lin)
        df_diff = model_spline.df_model - model_lin.df_model
        p_lrt = stats.chi2.sf(lr_stat, df_diff)
        reject_H0 = p_lrt < alpha

        lrt_res = {
            "lr_stat": lr_stat,
            "df_diff": df_diff,
            "p_value": p_lrt,
            "reject_H0": reject_H0
        }

        x_grid = np.linspace(df[var].min(), df[var].max(), 200)
        y_name = formula_base.split("~")[0].strip()

        rhs = formula_base.split("~")[1]
        raw_terms = [t.strip() for t in rhs.split("+") if t.strip() != ""]

        new_df = pd.DataFrame({var: x_grid})

        for term in raw_terms:
            if term == var or term == f"bs({var}, df={df_spline})":
                continue

            if term.startswith("C("):
                inside = term[2:].strip("() ")
                var_name = inside.split(",")[0].strip()
                if var_name not in df.columns:
                    continue
                new_df[var_name] = df[var_name].mode()[0]

            elif term in df.columns:
                if df[term].dtype.kind in "bifc":
                    if at == "mean":
                        new_df[term] = df[term].mean()
                    elif at == "median":
                        new_df[term] = df[term].median()
                    else:
                        raise ValueError("at doit être 'mean' ou 'median'.")
                else:
                    new_df[term] = df[term].mode()[0]
            else:
                continue

        y_hat_lin = model_lin.predict(new_df)
        y_hat_spl = model_spline.predict(new_df)

        grid = pd.DataFrame({
            var: x_grid,
            "y_hat_lin": y_hat_lin,
            "y_hat_spline": y_hat_spl
        })

        results[var] = {
            "model_lin": model_lin,
            "model_spline": model_spline,
            "lrt": lrt_res,
            "grid": grid
        }

    # Partie plot : grille 3 par ligne
    if plot and len(vars_list) > 0:
        n_vars = len(vars_list)
        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for i, var in enumerate(vars_list):
            r = i // n_cols
            c = i % n_cols
            ax = axes[r, c]

            grid = results[var]["grid"]
            lrt = results[var]["lrt"]

            ax.plot(grid[var], grid["y_hat_spline"], label=f"Spline ({var})", color="C0")
            ax.plot(grid[var], grid["y_hat_lin"], label="Linéaire", color="C1", linestyle="--")
            ax.set_xlabel(var)
            ax.set_ylabel(f"Pr({y_name}=1)")
            ax.set_title(f"{var} (p_LRT={lrt['p_value']:.3g})")
            ax.grid(True)
            ax.legend()

        # Masquer les axes vides si vars_list n'est pas un multiple de 3
        for j in range(i + 1, n_rows * n_cols):
            r = j // n_cols
            c = j % n_cols
            axes[r, c].axis("off")

        plt.tight_layout()
        plt.show()

    return results


# vars_to_test = [
#     "size",
#     "nb_rooms",
#     "nb_photos",
#     "revenu_median",
#     "taux_criminalite_1000",
#     "densite_services_rayon"
# ]

# res_all = spline_test_multi(
#     df=data_econometrie_scaled,
#     formula_base=equation,
#     vars_list=vars_to_test,
#     df_spline=4,
#     at="mean",
#     plot=True,
#     alpha=0.05
# )

# # Exemple : voir les résultats pour 'size'
# print(res_all["size"]["lrt"])
# res_all["size"]["grid"].head()


import pandas as pd

def add_centered_quadratic(df, vars_list):
    """
    Ajoute des termes quadratiques centrés pour une ou plusieurs variables.

    Pour chaque variable v de vars_list, crée :
        v_2_c = (v - mean(v))**2

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame à modifier (modifié en place).
    vars_list : str ou list of str
        Variable ou liste de variables continues.

    Retour
    ------
    df : pd.DataFrame
        Le DataFrame (même objet) avec les nouvelles colonnes ajoutées.
    """

    # Accepter un seul nom de variable en str
    if isinstance(vars_list, str):
        vars_list = [vars_list]

    created_cols = []

    for v in vars_list:
        if v not in df.columns:
            raise ValueError(f"La variable '{v}' n'existe pas dans le DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[v]):
            raise TypeError(f"La variable '{v}' doit être numérique pour créer un terme quadratique.")

        mean_v = df[v].mean()
        new_col = f"{v}_2_c"
        df[new_col] = (df[v] - mean_v) ** 2
        created_cols.append(new_col)

    if created_cols:
        print("Variables transformées (quadratique centrée) :")
        for v, c in zip(vars_list, created_cols):
            print(f"  - {v}  ->  {c}")

    return df



from patsy import bs  # nécessaire pour que la formule trouve bs()[web:154]

def add_spline_terms_to_formula(formula_base, vars_list, df_spline=4):
    """
    Ajoute des termes de spline B-splines (bs()) dans une formule Patsy
    pour une ou plusieurs variables continues, en conservant la structure
    du modèle de base.

    Exemple :
        base : "target ~ size + nb_rooms + revenu_median"
        vars_list = ["size", "revenu_median"]
        -> "target ~ bs(size, df=4) + nb_rooms + bs(revenu_median, df=4)"

    Paramètres
    ----------
    formula_base : str
        Formule de départ (sans splines) au format Patsy.
    vars_list : str ou list of str
        Variable ou liste de variables continues à transformer en splines.
    df_spline : int
        Degrés de liberté pour chaque spline (df>=3 recommandé).[web:147]

    Retour
    ------
    formula_spline : str
        Nouvelle formule avec bs(var, df=df_spline) injectés.
    """

    if isinstance(vars_list, str):
        vars_list = [vars_list]

    formula_spline = formula_base

    for var in vars_list:
        if var not in formula_base:
            raise ValueError(f"La variable '{var}' doit apparaître dans 'formula_base'.")

        # patrons classiques autour de var
        replace_patterns = [
            f"+ {var} +",
            f"+ {var}",
            f"{var} +",
            f"~ {var} +",
            f"~ {var}"
        ]

        replaced = False
        for pat in replace_patterns:
            if pat in formula_spline:
                formula_spline = formula_spline.replace(
                    pat,
                    pat.replace(var, f"bs({var}, df={df_spline})")
                )
                replaced = True

        if not replaced:
            raise ValueError(
                f"Impossible d'injecter la spline pour '{var}' proprement. "
                f"Vérifie que '{var}' apparaît de façon simple (pas uniquement transformée)."
            )

    print("Variables transformées en splines B-splines :")
    for v in vars_list:
        print(f"  - {v} -> bs({v}, df={df_spline})")

    return formula_spline


# equation_base = """
# target ~ approximate_latitude + approximate_longitude + size + nb_rooms
#         + nb_photos + revenu_median + taux_criminalite_1000
#         + densite_services_rayon + est_attrait_touristique
#         + C(property_type, Treatment(reference='appartement'))
#         + C(taille_agglomeration, Treatment(reference='200k - 2M hab'))
# """

# # Ajouter des splines sur size et revenu_median
# equation_spline = add_spline_terms_to_formula(
#     formula_base=equation_base,
#     vars_list=["size", "revenu_median"],
#     df_spline=4
# )

# print(equation_spline)

# # Fit logit avec splines
# logit_spline = smf.logit(equation_spline, data=data_econometrie_scaled).fit()




def add_centered_vars(df, vars_list):
    """
    Ajoute des versions centrées pour une ou plusieurs variables continues.

    Pour chaque variable v de vars_list, crée :
        v_c = v - mean(v)

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame à modifier (modifié en place).
    vars_list : str ou list of str
        Variable ou liste de variables continues.

    Retour
    ------
    df : pd.DataFrame
        Le DataFrame (même objet) avec les nouvelles colonnes ajoutées.
    """

    if isinstance(vars_list, str):
        vars_list = [vars_list]

    created_cols = []

    for v in vars_list:
        if v not in df.columns:
            raise ValueError(f"La variable '{v}' n'existe pas dans le DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[v]):
            raise TypeError(f"La variable '{v}' doit être numérique pour être centrée.")

        mean_v = df[v].mean()
        new_col = f"{v}_c"
        df[new_col] = df[v] - mean_v
        created_cols.append(new_col)

    if created_cols:
        print("Variables centrées créées :")
        for v, c in zip(vars_list, created_cols):
            print(f"  - {v}  ->  {c}")

    return df


