"""
Utility functions to load and manage Keras Tuner hyperparameter configurations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def load_keras_tuner_config(config_path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a Keras Tuner trial config JSON.
    
    Parameters
    ----------
    config_path : str
        Path to the trial config JSON file from Keras Tuner
        Usually at: <tuner_dir>/<project_name>/trial_<N>/trial.json
        
    Returns
    -------
    dict
        Hyperparameters dictionary
        
    Example
    -------
    >>> config = load_keras_tuner_config('my_dir/helloworld/trial_0/trial.json')
    >>> print(config)
    {'units': 256, 'dropout': 0.2, 'optimizer': 'adam'}
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract hyperparameters from the config
    values = config.get('hyperparameters', {}).get('values', {})
    return values if values else config.get('hyperparameters', {})


def display_keras_tuner_best_trial(tuner_dir: str, project_name: str) -> Optional[Dict[str, Any]]:
    """
    Load and display the best trial's hyperparameters from a Keras Tuner search.
    
    Parameters
    ----------
    tuner_dir : str
        Directory where tuner results are stored
    project_name : str
        Project name used in tuner
        
    Returns
    -------
    dict or None
        Best hyperparameters if found, None otherwise
        
    Example
    -------
    >>> best_hp = display_keras_tuner_best_trial('my_dir', 'helloworld')
    >>> print(best_hp)
    {'units': 256, 'dropout': 0.2}
    """
    tuner_path = Path(tuner_dir) / project_name
    
    if not tuner_path.exists():
        print(f"❌ Tuner directory not found: {tuner_path}")
        return None
    
    # Find all trial directories
    trial_dirs = sorted([d for d in tuner_path.iterdir() if d.is_dir() and d.name.startswith('trial_')])
    
    if not trial_dirs:
        print(f"❌ No trials found in {tuner_path}")
        return None
    
    print(f"Found {len(trial_dirs)} trials\n")
    
    # Load metadata if exists
    metadata_file = tuner_path / 'metadata.json'
    best_trial = None
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            best_trial = metadata.get('best_trial', {})
            print(f"📊 Best Trial Info: {best_trial}")
            print()
    
    # Display all trials
    trials_data = []
    for trial_dir in trial_dirs:
        trial_json = trial_dir / 'trial.json'
        if trial_json.exists():
            with open(trial_json, 'r') as f:
                trial = json.load(f)
                
            hp_values = trial.get('hyperparameters', {}).get('values', {})
            score = trial.get('score', 'N/A')
            status = trial.get('status', 'N/A')
            
            trials_data.append({
                'Trial': trial_dir.name,
                'Score': score,
                'Status': status,
                'Hyperparameters': str(hp_values)[:80] + '...' if len(str(hp_values)) > 80 else str(hp_values)
            })
    
    if trials_data:
        df_trials = pd.DataFrame(trials_data)
        print("📋 Summary of All Trials:")
        print(df_trials.to_string(index=False))
        print()
    
    # Find best trial's hyperparameters
    best_trial_file = None
    best_score = None
    
    for trial_dir in trial_dirs:
        trial_json = trial_dir / 'trial.json'
        if trial_json.exists():
            with open(trial_json, 'r') as f:
                trial = json.load(f)
            
            score = trial.get('score')
            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score
                    best_trial_file = trial_json
    
    if best_trial_file:
        with open(best_trial_file, 'r') as f:
            best_config = json.load(f)
        
        hp_values = best_config.get('hyperparameters', {}).get('values', {})
        
        print(f"🏆 Best Trial Hyperparameters (Score: {best_score}):")
        print("=" * 70)
        for param, value in hp_values.items():
            print(f"  {param:25s}: {value}")
        print("=" * 70)
        
        return hp_values
    
    return None


def load_all_trials_as_dataframe(tuner_dir: str, project_name: str) -> Optional[pd.DataFrame]:
    """
    Load all trials from a Keras Tuner search as a pandas DataFrame.
    
    Parameters
    ----------
    tuner_dir : str
        Directory where tuner results are stored
    project_name : str
        Project name used in tuner
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with trial results and hyperparameters
        
    Example
    -------
    >>> df = load_all_trials_as_dataframe('my_dir', 'helloworld')
    >>> print(df.head())
    """
    tuner_path = Path(tuner_dir) / project_name
    
    if not tuner_path.exists():
        print(f"❌ Tuner directory not found: {tuner_path}")
        return None
    
    trial_dirs = sorted([d for d in tuner_path.iterdir() if d.is_dir() and d.name.startswith('trial_')])
    
    if not trial_dirs:
        print(f"❌ No trials found in {tuner_path}")
        return None
    
    all_trials = []
    
    for trial_dir in trial_dirs:
        trial_json = trial_dir / 'trial.json'
        if trial_json.exists():
            with open(trial_json, 'r') as f:
                trial = json.load(f)
            
            hp_values = trial.get('hyperparameters', {}).get('values', {})
            row = {
                'trial_id': trial_dir.name,
                'score': trial.get('score', None),
                'status': trial.get('status', 'unknown'),
                **hp_values  # Flatten hyperparameters as columns
            }
            all_trials.append(row)
    
    df = pd.DataFrame(all_trials)
    return df.sort_values('score', ascending=False, na_position='last')


def create_model_with_best_hyperparameters(build_model_fn, tuner_dir: str, project_name: str):
    """
    Create a model using the best hyperparameters found by Keras Tuner.
    
    Parameters
    ----------
    build_model_fn : callable
        Function that builds model given hyperparameters
    tuner_dir : str
        Directory where tuner results are stored
    project_name : str
        Project name used in tuner
        
    Returns
    -------
    keras.Model or None
        Compiled model with best hyperparameters, or None if load fails
        
    Example
    -------
    >>> def build_model(hp):
    ...     model = Sequential()
    ...     model.add(Dense(hp['units'], activation='relu'))
    ...     return model
    >>> 
    >>> best_model = create_model_with_best_hyperparameters(
    ...     build_model, 'my_dir', 'helloworld'
    ... )
    """
    best_hp = display_keras_tuner_best_trial(tuner_dir, project_name)
    
    if best_hp is None:
        print("❌ Could not load best hyperparameters")
        return None
    
    print("\n🔨 Creating model with best hyperparameters...")
    
    # Convert dict to HyperParameters-like object if needed
    try:
        model = build_model_fn(best_hp)
        print("✅ Model created successfully!")
        return model
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None
