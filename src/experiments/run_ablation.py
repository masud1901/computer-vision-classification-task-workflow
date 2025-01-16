"""Run ablation study experiments."""

import os
import pandas as pd
from src.models.ablation_study import AblationStudy
from src.training.trainer import ModelTrainer
from src.data.data_loader import DataLoader
from src.config import CONFIG


def run_ablation_experiments(config):
    """Run ablation study experiments and compare results."""
    # Initialize data loader
    data_loader = DataLoader(config)
    train_generator, val_generator, test_generator = data_loader.create_generators()
    
    # Initialize ablation study
    ablation = AblationStudy(
        input_shape=(*config['image_size'], 3),
        num_classes=len(train_generator.class_indices)
    )
    
    # Get all model variants
    models = ablation.run_ablation_study()
    
    # Results storage
    results = []
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Initialize trainer
        trainer = ModelTrainer(model, config)
        trainer.compile_model()
        
        # Train model
        history = trainer.train(
            train_generator,
            val_generator,
            class_weights=data_loader.get_class_weights()
        )
        
        # Evaluate model
        metrics = trainer.evaluate(test_generator)
        
        # Store results
        results.append({
            'model': model_name,
            'test_accuracy': metrics['accuracy'],
            'test_loss': metrics['loss']
        })
        
        # Save model
        model.save(f'models/ablation_{model_name}.h5')
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/ablation_results.csv', index=False)
    
    return results_df


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run experiments
    results = run_ablation_experiments(CONFIG)
    
    # Print results
    print("\nAblation Study Results:")
    print(results.to_string(index=False)) 