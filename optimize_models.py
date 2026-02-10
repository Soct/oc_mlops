"""
Script d'optimisation de modèles avec Optuna et MLflow.

Ce script offre une alternative au notebook pour une exécution automatisée.
Idéal pour lancer des optimisations en arrière-plan ou sur un serveur.

Usage:
    # Configuration rapide (20 trials)
    python optimize_models.py --preset quick_test
    
    # Configuration standard (50 trials)
    python optimize_models.py --preset standard
    
    # Configuration personnalisée
    python optimize_models.py --model lightgbm --trials 100 --metric business_cost
    
    # Optimiser tous les modèles
    python optimize_models.py --preset deep_search --cv-folds 5
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports ML
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, 
    make_scorer, confusion_matrix
)

import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow

# Configuration locale
from config_optuna import get_config, MODEL_CONFIGS, METRICS_INFO


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
DATA_PATH = Path("data/features_engineered.parquet")


# ============================================================================
# MÉTRIQUES
# ============================================================================

def business_cost_scorer(y_true, y_pred):
    """Coût métier : FN coûte 10 fois plus cher que FP."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = fp * 1 + fn * 10
    return -cost


# ============================================================================
# PIPELINES
# ============================================================================

def create_xgboost_pipeline(params, y_train):
    """Créer un pipeline XGBoost."""
    default_params = {
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    default_params.update(params)
    default_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', xgb.XGBClassifier(**default_params))
    ])


def create_lightgbm_pipeline(params):
    """Créer un pipeline LightGBM."""
    default_params = {
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'is_unbalance': True
    }
    default_params.update(params)
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(**default_params))
    ])


def create_mlp_pipeline(params):
    """Créer un pipeline MLP."""
    default_params = {
        'solver': 'adam',
        'max_iter': 300,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': RANDOM_STATE,
        'verbose': False
    }
    default_params.update(params)
    
    return Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', MLPClassifier(**default_params))
    ])


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def load_data():
    """Charger et préparer les données."""
    print(f"📂 Chargement des données: {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    
    # Séparer train/test
    train = df[df['TARGET'].notna()].copy()
    test = df[df['TARGET'].isna()].copy()
    
    # Features et target
    X_train = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y_train = train['TARGET']
    X_test = test.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    test_ids = test['SK_ID_CURR']
    
    # Nettoyage
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"✅ Données chargées:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Classe 0: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"   Classe 1: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
    
    return X_train, y_train, X_test, test_ids


# ============================================================================
# OPTIMISATION OPTUNA
# ============================================================================

def create_objective(model_type, X_train, y_train, skf, scoring, metric):
    """Créer la fonction objectif pour Optuna."""
    
    def objective(trial):
        # Suggérer les hyperparamètres
        search_space = MODEL_CONFIGS[model_type]["search_space"]
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_name == "n_units_range":
                continue  # Géré séparément pour MLP
            
            param_type = param_config["type"]
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
        
        # Cas spécial MLP : couches cachées
        if model_type == "mlp" and "n_layers" in params:
            n_layers = params.pop("n_layers")
            hidden_layers = []
            units_config = search_space["n_units_range"]
            for i in range(n_layers):
                hidden_layers.append(
                    trial.suggest_int(
                        f'n_units_l{i}',
                        units_config["low"],
                        units_config["high"],
                        step=units_config.get("step", 1)
                    )
                )
            params["hidden_layer_sizes"] = tuple(hidden_layers)
        
        # Créer le pipeline
        if model_type == "xgboost":
            pipeline = create_xgboost_pipeline(params, y_train)
        elif model_type == "lightgbm":
            pipeline = create_lightgbm_pipeline(params)
        elif model_type == "mlp":
            pipeline = create_mlp_pipeline(params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Validation croisée
        try:
            cv_results = cross_validate(
                pipeline, X_train, y_train,
                cv=skf, scoring=scoring,
                n_jobs=1, return_train_score=False,
                error_score='raise'
            )
            return np.mean(cv_results[f'test_{metric}'])
        except Exception as e:
            print(f"⚠️ Erreur trial: {e}")
            return -np.inf if metric == 'business_cost' else 0.0
    
    return objective


def optimize_model(model_type, X_train, y_train, config):
    """Optimiser un modèle avec Optuna."""
    
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMISATION: {model_type.upper()}")
    print(f"{'='*80}")
    
    # CV et scoring
    skf = StratifiedKFold(
        n_splits=config['cv_folds'],
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    scoring = {
        'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
        'recall_minority': make_scorer(recall_score, pos_label=1, zero_division=0),
        'f1': make_scorer(f1_score, pos_label=1, zero_division=0),
        'business_cost': make_scorer(business_cost_scorer)
    }
    
    # Créer l'étude Optuna
    study = optuna.create_study(
        study_name=f"{model_type}_{config['metric']}",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    # MLflow callback
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=config['metric'],
        create_experiment=False,
        mlflow_kwargs={
            "experiment_id": mlflow.get_experiment_by_name(
                "Advanced Models - Optuna Optimization"
            ).experiment_id,
            "nested": True
        }
    )
    
    # Parent run MLflow
    with mlflow.start_run(run_name=f"{model_type.upper()} - Optuna {config['n_trials']} trials"):
        
        # Tags
        mlflow.set_tags({
            "author": "Automated Optimization",
            "project": "Home Credit Default Risk",
            "phase": "optimization",
            "model_type": model_type,
            "optimizer": "optuna",
            "environment": "development"
        })
        
        mlflow.log_params({
            "model_type": model_type,
            "n_trials": config['n_trials'],
            "metric": config['metric'],
            "cv_folds": config['cv_folds'],
            "n_samples": len(X_train),
            "n_features": X_train.shape[1]
        })
        
        # Optimiser
        objective_fn = create_objective(
            model_type, X_train, y_train, skf, scoring, config['metric']
        )
        
        study.optimize(
            objective_fn,
            n_trials=config['n_trials'],
            timeout=config.get('timeout'),
            callbacks=[mlflow_callback],
            show_progress_bar=True
        )
        
        # Résultats
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n✅ OPTIMISATION TERMINÉE")
        print(f"   Meilleur {config['metric']}: {best_value:.4f}")
        
        # Logger les meilleurs résultats
        mlflow.log_metric(f"best_{config['metric']}", best_value)
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)
        
        # Entraîner le modèle final
        print(f"\n📦 Entraînement du modèle final...")
        
        if model_type == "xgboost":
            best_pipeline = create_xgboost_pipeline(best_params, y_train)
        elif model_type == "lightgbm":
            best_pipeline = create_lightgbm_pipeline(best_params)
        elif model_type == "mlp":
            best_pipeline = create_mlp_pipeline(best_params)
        
        # CV finale
        final_cv = cross_validate(
            best_pipeline, X_train, y_train,
            cv=skf, scoring=scoring,
            n_jobs=1, return_train_score=False
        )
        
        # Logger toutes les métriques
        for metric_name in scoring.keys():
            mean_val = np.mean(final_cv[f'test_{metric_name}'])
            std_val = np.std(final_cv[f'test_{metric_name}'])
            mlflow.log_metric(f"{metric_name}_mean", mean_val)
            mlflow.log_metric(f"{metric_name}_std", std_val)
        
        # Entraîner et sauvegarder
        best_pipeline.fit(X_train, y_train)
        
        signature = mlflow.models.signature.infer_signature(
            X_train,
            best_pipeline.predict_proba(X_train)[:, 1]
        )
        
        mlflow.sklearn.log_model(
            best_pipeline,
            "model",
            signature=signature,
            input_example=X_train.head(3)
        )
    
    return best_params, best_value, study, best_pipeline


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimisation de modèles avec Optuna et MLflow"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick_test", "standard", "deep_search", "business_cost"],
        help="Configuration prédéfinie"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm", "mlp", "all"],
        default="all",
        help="Modèle à optimiser"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        help="Nombre de trials Optuna"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        choices=list(METRICS_INFO.keys()),
        help="Métrique à optimiser"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Nombre de folds pour la validation croisée"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout en secondes"
    )
    
    args = parser.parse_args()
    
    # Déterminer la configuration
    if args.preset:
        config = get_config(args.preset)
    else:
        config = get_config("standard")  # Par défaut
    
    # Override avec les arguments
    if args.trials:
        config['n_trials'] = args.trials
    if args.metric:
        config['metric'] = args.metric
    if args.cv_folds:
        config['cv_folds'] = args.cv_folds
    if args.timeout:
        config['timeout'] = args.timeout
    
    # Déterminer les modèles à optimiser
    if args.model == "all":
        models_to_run = config.get('models_to_run', ["xgboost", "lightgbm", "mlp"])
    else:
        models_to_run = [args.model]
    
    # Afficher la configuration
    print(f"\n{'='*80}")
    print(f"🚀 OPTIMISATION DE MODÈLES")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Modèles: {', '.join(models_to_run)}")
    print(f"  Trials: {config['n_trials']}")
    print(f"  Métrique: {config['metric']}")
    print(f"  CV Folds: {config['cv_folds']}")
    print(f"  Timeout: {config.get('timeout', 'Aucun')}")
    
    # Configurer MLflow
    import os
    tracking_uri = os.path.abspath(os.path.join(os.getcwd(), 'mlruns'))
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    mlflow.set_experiment("Advanced Models - Optuna Optimization")
    
    print(f"\n📁 MLflow URI: {mlflow.get_tracking_uri()}")
    
    # Charger les données
    X_train, y_train, X_test, test_ids = load_data()
    
    # Optimiser chaque modèle
    results = {}
    
    for model_type in models_to_run:
        if not MODEL_CONFIGS[model_type]["enabled"]:
            print(f"\n⚠️ {model_type} est désactivé dans la configuration")
            continue
        
        try:
            params, score, study, pipeline = optimize_model(
                model_type, X_train, y_train, config
            )
            results[model_type] = {
                'params': params,
                'score': score,
                'study': study,
                'pipeline': pipeline
            }
        except Exception as e:
            print(f"\n❌ Erreur lors de l'optimisation de {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Résumé final
    print(f"\n{'='*80}")
    print(f"✅ OPTIMISATION TERMINÉE")
    print(f"{'='*80}\n")
    
    if results:
        comparison = pd.DataFrame({
            'Model': list(results.keys()),
            f'Best {config["metric"]}': [r['score'] for r in results.values()]
        }).sort_values(f'Best {config["metric"]}', ascending=False)
        
        print(comparison.to_string(index=False))
        
        best_model = comparison.iloc[0]['Model']
        best_score = comparison.iloc[0][f'Best {config["metric"]}']
        
        print(f"\n🏆 Meilleur modèle: {best_model.upper()}")
        print(f"   Score: {best_score:.4f}")
        
        # Sauvegarder le meilleur modèle
        import joblib
        model_filename = f'best_model_{best_model}.pkl'
        joblib.dump(results[best_model]['pipeline'], model_filename)
        print(f"\n💾 Modèle sauvegardé: {model_filename}")
        
        # Générer les prédictions
        test_pred = results[best_model]['pipeline'].predict_proba(X_test)[:, 1]
        submission = pd.DataFrame({
            'SK_ID_CURR': test_ids,
            'TARGET': test_pred
        })
        submission_filename = f'submission_{best_model}.csv'
        submission.to_csv(submission_filename, index=False)
        print(f"📊 Soumission créée: {submission_filename}")
    else:
        print("❌ Aucun modèle optimisé avec succès")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
