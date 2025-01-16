"""Model training and evaluation utilities with advanced monitoring."""

import os
import logging
from typing import Optional, Dict, Any

import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau,
    CSVLogger,
    Callback
)
import datetime
from src.visualization.visualizer import Visualizer


class TrainingMonitor(Callback):
    """Custom callback for detailed training monitoring."""
    
    def __init__(self, logger):
        super(TrainingMonitor, self).__init__()
        self.logger = logger
        self.epoch_times = []
        self.best_val_acc = 0
        self._start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self._start_time = datetime.datetime.now()
        self.logger.info(f"\nStarting epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = datetime.datetime.now() - self._start_time
        self.epoch_times.append(epoch_time.total_seconds())
        
        # Log metrics
        metrics_str = " - ".join([
            f"{k}: {v:.4f}" for k, v in logs.items()
        ])
        self.logger.info(f"Epoch {epoch + 1}: {metrics_str}")
        
        # Track best performance
        if logs.get('val_accuracy', 0) > self.best_val_acc:
            self.best_val_acc = logs.get('val_accuracy')
            self.logger.info(
                f"New best validation accuracy: {self.best_val_acc:.4f}"
            )


class ModelTrainer:
    """Advanced model trainer with performance monitoring and optimization."""

    def __init__(self, model, config: dict):
        """Initialize the trainer.
        
        Args:
            model: Keras model instance
            config (dict): Training configuration
        """
        self.model = model
        self.config = config
        self.history = None
        self.visualizer = Visualizer()
        self.logger = self._setup_logger()
        self.monitor = TrainingMonitor(self.logger)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('logs/training.log')
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger

    def compile_model(self, learning_rate: float = 0.001) -> None:
        """Compile the model with specified optimizer and loss function.
        
        Args:
            learning_rate (float): Initial learning rate
        """
        try:
            self.model.compile(
                optimizer=self.config.get(
                    'optimizer', 
                    'adam'
                ),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.logger.info("Model compiled successfully")
        except Exception as e:
            self.logger.error(f"Error compiling model: {str(e)}")
            raise

    def _create_callbacks(self) -> list:
        """Create training callbacks with proper configuration."""
        callbacks_dir = 'models'
        logs_dir = 'logs'
        os.makedirs(callbacks_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        callbacks = [
            self.monitor,
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(callbacks_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=self.config.get('reduce_lr_patience', 5),
                min_lr=1e-6,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(
                    logs_dir,
                    'fit',
                    datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                ),
                histogram_freq=1,
                update_freq='epoch'
            ),
            CSVLogger(
                os.path.join(logs_dir, 'training_history.csv'),
                append=True
            )
        ]
        return callbacks

    def train(
        self,
        train_generator,
        val_generator,
        class_weights: Optional[Dict[int, float]] = None
    ) -> None:
        """Train the model with advanced monitoring.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            class_weights: Optional class weights for imbalanced datasets
        """
        try:
            self.logger.info("Starting model training...")
            
            callbacks = self._create_callbacks()
            
            self.history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.get('epochs', 50),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Plot training history
            self.visualizer.plot_training_history(self.history)
            
            # Log training summary
            self._log_training_summary()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def _log_training_summary(self) -> None:
        """Log training summary statistics."""
        avg_epoch_time = np.mean(self.monitor.epoch_times)
        self.logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
        self.logger.info(f"Best validation accuracy: {self.monitor.best_val_acc:.4f}")

    def evaluate(self, test_generator) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        try:
            self.logger.info("Evaluating model on test set...")
            
            # Evaluate model
            test_results = self.model.evaluate(test_generator)
            metrics = dict(zip(self.model.metrics_names, test_results))
            
            # Log metrics
            for metric_name, value in metrics.items():
                self.logger.info(f"Test {metric_name}: {value:.4f}")
            
            # Generate predictions
            predictions = self.model.predict(test_generator)
            y_pred = np.argmax(predictions, axis=1)
            y_true = test_generator.classes
            
            # Plot confusion matrix
            self.visualizer.plot_confusion_matrix(
                y_true,
                y_pred,
                test_generator.class_indices
            )
            
            # Plot ROC curves
            self.visualizer.plot_roc_curves(
                test_generator=test_generator,
                model=self.model,
                true_labels=y_true,
                pred_probs=predictions
            )
            
            # Plot metrics summary
            self.visualizer.plot_metrics_summary(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            self.model.save(filepath)
            self.logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise