"""
Configuration pour l'optimisation des modèles avancés.

Ce fichier permet de centraliser tous les paramètres d'optimisation
sans avoir à modifier le notebook principal.

Usage:
    from config_optuna import OPTUNA_CONFIG, MODEL_CONFIGS
"""

# ============================================================================
# CONFIGURATION GÉNÉRALE
# ============================================================================

OPTUNA_CONFIG = {
    # Nombre de trials par modèle
    "n_trials": {
        "quick": 20,      # Test rapide (~15-20 min total)
        "normal": 50,     # Optimisation normale (~45-60 min total)
        "deep": 100,      # Optimisation approfondie (~2-3h total)
        "extensive": 200  # Recherche extensive (~4-6h total)
    },
    
    # Métrique à optimiser
    "metric": "roc_auc",  # Options: 'roc_auc', 'f1', 'recall_minority', 'business_cost'
    
    # Validation croisée
    "cv_folds": 3,  # Nombre de folds (3-5 recommandé)
    
    # Timeout (optionnel, en secondes)
    "timeout": None,  # Ex: 1800 pour 30 minutes max par modèle
    
    # Random seed pour reproductibilité
    "random_state": 42,
    
    # Sampler Optuna
    "sampler": "TPE",  # Options: 'TPE', 'Random', 'Grid', 'CmaEs'
    
    # Pruner (arrêt précoce des mauvais trials)
    "use_pruner": False,
    "pruner_config": {
        "n_startup_trials": 5,
        "n_warmup_steps": 10
    }
}


# ============================================================================
# ESPACES DE RECHERCHE PAR MODÈLE
# ============================================================================

MODEL_CONFIGS = {
    
    # ========================================================================
    # XGBOOST
    # ========================================================================
    "xgboost": {
        "enabled": True,
        "search_space": {
            # Nombre d'arbres
            "n_estimators": {
                "type": "int",
                "low": 50,
                "high": 300,
                "step": 50
            },
            
            # Profondeur maximale
            "max_depth": {
                "type": "int",
                "low": 3,
                "high": 10
            },
            
            # Taux d'apprentissage
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True
            },
            
            # Échantillonnage des observations
            "subsample": {
                "type": "float",
                "low": 0.6,
                "high": 1.0
            },
            
            # Échantillonnage des features
            "colsample_bytree": {
                "type": "float",
                "low": 0.6,
                "high": 1.0
            },
            
            # Poids minimum des enfants
            "min_child_weight": {
                "type": "int",
                "low": 1,
                "high": 10
            },
            
            # Réduction minimale de perte pour split
            "gamma": {
                "type": "float",
                "low": 0,
                "high": 5
            },
            
            # Régularisation L1
            "reg_alpha": {
                "type": "float",
                "low": 0,
                "high": 10
            },
            
            # Régularisation L2
            "reg_lambda": {
                "type": "float",
                "low": 0,
                "high": 10
            }
        },
        
        # Paramètres fixes (non optimisés)
        "fixed_params": {
            "random_state": 42,
            "n_jobs": -1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0
        }
    },
    
    # ========================================================================
    # LIGHTGBM
    # ========================================================================
    "lightgbm": {
        "enabled": True,
        "search_space": {
            # Nombre d'arbres
            "n_estimators": {
                "type": "int",
                "low": 50,
                "high": 300,
                "step": 50
            },
            
            # Profondeur maximale (-1 = illimité)
            "max_depth": {
                "type": "int",
                "low": 3,
                "high": 15
            },
            
            # Taux d'apprentissage
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True
            },
            
            # Nombre de feuilles
            "num_leaves": {
                "type": "int",
                "low": 20,
                "high": 150
            },
            
            # Échantillonnage des observations
            "subsample": {
                "type": "float",
                "low": 0.6,
                "high": 1.0
            },
            
            # Échantillonnage des features
            "colsample_bytree": {
                "type": "float",
                "low": 0.6,
                "high": 1.0
            },
            
            # Échantillons minimum par feuille
            "min_child_samples": {
                "type": "int",
                "low": 5,
                "high": 50
            },
            
            # Régularisation L1
            "reg_alpha": {
                "type": "float",
                "low": 0,
                "high": 10
            },
            
            # Régularisation L2
            "reg_lambda": {
                "type": "float",
                "low": 0,
                "high": 10
            }
        },
        
        # Paramètres fixes
        "fixed_params": {
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "is_unbalance": True
        }
    },
    
    # ========================================================================
    # MLP (Multi-Layer Perceptron)
    # ========================================================================
    "mlp": {
        "enabled": True,
        "search_space": {
            # Nombre de couches cachées
            "n_layers": {
                "type": "int",
                "low": 1,
                "high": 3
            },
            
            # Neurones par couche (défini dynamiquement)
            "n_units_range": {
                "type": "int",
                "low": 50,
                "high": 200,
                "step": 50
            },
            
            # Fonction d'activation
            "activation": {
                "type": "categorical",
                "choices": ["relu", "tanh"]
            },
            
            # Régularisation L2
            "alpha": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True
            },
            
            # Taux d'apprentissage initial
            "learning_rate_init": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True
            }
        },
        
        # Paramètres fixes
        "fixed_params": {
            "solver": "adam",
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": 42,
            "verbose": False
        }
    }
}


# ============================================================================
# CONFIGURATIONS PRÉDÉFINIES
# ============================================================================

PRESET_CONFIGS = {
    # Configuration pour un test rapide
    "quick_test": {
        "n_trials_mode": "quick",
        "cv_folds": 3,
        "models_to_run": ["lightgbm"],  # Le plus rapide
        "timeout": 600,  # 10 minutes max
    },
    
    # Configuration standard
    "standard": {
        "n_trials_mode": "normal",
        "cv_folds": 3,
        "models_to_run": ["xgboost", "lightgbm"],
        "timeout": None,
    },
    
    # Configuration approfondie
    "deep_search": {
        "n_trials_mode": "deep",
        "cv_folds": 5,
        "models_to_run": ["xgboost", "lightgbm", "mlp"],
        "timeout": None,
    },
    
    # Optimisation du coût métier
    "business_cost": {
        "n_trials_mode": "normal",
        "cv_folds": 3,
        "models_to_run": ["xgboost", "lightgbm"],
        "metric": "business_cost",
        "timeout": None,
    }
}


# ============================================================================
# MÉTRIQUES DISPONIBLES
# ============================================================================

METRICS_INFO = {
    "roc_auc": {
        "name": "ROC-AUC",
        "description": "Area Under the ROC Curve",
        "direction": "maximize",
        "range": [0, 1],
        "best_for": "Classification déséquilibrée générale"
    },
    
    "f1": {
        "name": "F1-Score",
        "description": "Harmonic mean of precision and recall",
        "direction": "maximize",
        "range": [0, 1],
        "best_for": "Équilibre précision/recall"
    },
    
    "recall_minority": {
        "name": "Recall (classe 1)",
        "description": "True Positive Rate",
        "direction": "maximize",
        "range": [0, 1],
        "best_for": "Minimiser les faux négatifs"
    },
    
    "business_cost": {
        "name": "Coût métier",
        "description": "FP × 1 + FN × 10 (négatif)",
        "direction": "maximize",  # Moins négatif = meilleur
        "range": [-float('inf'), 0],
        "best_for": "Optimisation selon coûts métier réels"
    }
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_config(preset="standard"):
    """
    Retourne une configuration prédéfinie.
    
    Args:
        preset: Nom du preset ('quick_test', 'standard', 'deep_search', 'business_cost')
    
    Returns:
        dict: Configuration complète
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Preset inconnu: {preset}. Choix: {list(PRESET_CONFIGS.keys())}")
    
    config = OPTUNA_CONFIG.copy()
    preset_config = PRESET_CONFIGS[preset]
    
    # Mettre à jour avec les valeurs du preset
    config["n_trials"] = OPTUNA_CONFIG["n_trials"][preset_config["n_trials_mode"]]
    config["cv_folds"] = preset_config.get("cv_folds", 3)
    config["timeout"] = preset_config.get("timeout")
    config["metric"] = preset_config.get("metric", "roc_auc")
    config["models_to_run"] = preset_config.get("models_to_run", ["xgboost", "lightgbm", "mlp"])
    
    return config


def print_config(preset="standard"):
    """Affiche la configuration dans un format lisible."""
    config = get_config(preset)
    
    print(f"\n{'='*80}")
    print(f"Configuration: {preset.upper()}")
    print(f"{'='*80}\n")
    
    print(f"Optimisation:")
    print(f"  Trials par modèle: {config['n_trials']}")
    print(f"  Métrique: {config['metric']}")
    print(f"  CV Folds: {config['cv_folds']}")
    print(f"  Timeout: {config['timeout'] or 'Aucun'}")
    print(f"  Sampler: {config['sampler']}")
    
    print(f"\nModèles activés:")
    for model in config['models_to_run']:
        enabled = MODEL_CONFIGS[model]['enabled']
        status = "✅" if enabled else "❌"
        print(f"  {status} {model.upper()}")
    
    print(f"\nTemps estimé:")
    n_models = len(config['models_to_run'])
    min_time = (config['n_trials'] * 10 * n_models) // 60
    max_time = (config['n_trials'] * 30 * n_models) // 60
    print(f"  {min_time}-{max_time} minutes")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Afficher toutes les configurations disponibles
    print("📋 CONFIGURATIONS PRÉDÉFINIES\n")
    
    for preset_name in PRESET_CONFIGS.keys():
        print_config(preset_name)
    
    # Exemple : Récupérer une config
    config = get_config("standard")
    print(f"Nombre de trials: {config['n_trials']}")
    print(f"Métrique: {config['metric']}")
