"""Main script to run the plant disease classifier with advanced configuration."""

import os
import logging
from pathlib import Path
import tensorflow as tf

from src.config import CONFIG
from src.data.data_loader import DataLoader
from src.models.cnn_model import PlantDiseaseModel
from src.training.trainer import ModelTrainer


def setup_logger() -> logging.Logger:
    """Setup main logger for the application."""
    logger = logging.getLogger('PlantDiseaseClassifier')
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('logs/main.log')

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def create_directories() -> None:
    """Create necessary directories for model artifacts."""
    directories = [
        'models',
        'logs',
        'plots',
        'logs/fit'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_gpu() -> None:
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth should be set before GPUs have been initialized
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {str(e)}")
    else:
        logger.warning("No GPUs found, using CPU")


def main() -> None:
    """Main execution function."""
    try:
        # Initial setup
        logger.info("Starting Plant Disease Classification training")
        create_directories()
        setup_gpu()

        # Initialize data loader
        logger.info("Initializing data loader")
        data_loader = DataLoader(CONFIG)
        data_loader.split_dataset()

        # Create data generators
        train_generator, val_generator, test_generator = data_loader.create_generators()

        # Calculate class weights for imbalanced dataset
        class_weights = data_loader.get_class_weights()
        steps_per_epoch = data_loader.get_steps_per_epoch()

        # Initialize model
        logger.info("Building model architecture")
        input_shape = (*CONFIG['image_size'], 3)
        num_classes = len(train_generator.class_indices)
        model_builder = PlantDiseaseModel(input_shape, num_classes)
        model = model_builder.build_model()

        # Initialize trainer
        logger.info("Setting up model trainer")
        trainer = ModelTrainer(model, CONFIG)
        trainer.compile_model(learning_rate=CONFIG.get('learning_rate', 0.001))
        # Train model
        logger.info("Starting model training")
        trainer.train(
            train_generator=train_generator,
            val_generator=val_generator,
            class_weights=class_weights
        )

        # Evaluate model
        logger.info("Evaluating model performance")
        metrics = trainer.evaluate(test_generator)

        # Save final model
        trainer.save_model('models/final_model.h5')
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    logger = setup_logger()
    main() 