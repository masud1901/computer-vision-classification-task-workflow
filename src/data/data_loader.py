"""Data loading and preprocessing utilities with optimized performance."""

import os
import logging
from typing import Tuple, Optional
import splitfolders
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import mobilenet_v2


class DataLoader:
    """Optimized data loader with advanced augmentation and preprocessing."""

    def __init__(self, config: dict):
        """Initialize DataLoader with configuration.
        
        Args:
            config (dict): Configuration dictionary containing data parameters
        """
        self.config = config
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the data loader."""
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def split_dataset(self) -> None:
        """Split dataset into train/val/test sets with error handling."""
        try:
            self.logger.info("Splitting dataset into train/val/test sets...")
            splitfolders.ratio(
                self.config['input_folder'],
                output=self.config['output_folder'],
                seed=42,
                ratio=(
                    self.config['train_split'],
                    self.config['val_split'],
                    self.config['test_split']
                )
            )
            self.logger.info("Dataset split completed successfully")
        except Exception as e:
            self.logger.error(f"Error splitting dataset: {str(e)}")
            raise

    def _create_augmentation_pipeline(self, training: bool = True) -> ImageDataGenerator:
        """Create data augmentation pipeline."""
        if training:
            return ImageDataGenerator(
                preprocessing_function=mobilenet_v2.preprocess_input,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest',
                validation_split=0.0
            )
        return ImageDataGenerator(
            preprocessing_function=mobilenet_v2.preprocess_input
        )

    def _create_dataset_generator(
        self,
        generator: ImageDataGenerator,
        subset: str,
        shuffle: bool = True
    ):
        """Create dataset generator.
        
        Args:
            generator: ImageDataGenerator instance
            subset: Dataset subset ('train', 'val', or 'test')
            shuffle: Whether to shuffle the data
            
        Returns:
            DirectoryIterator: Dataset generator
        """
        directory = os.path.join(self.config['output_folder'], subset)
        
        return generator.flow_from_directory(
            directory,
            target_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            class_mode='categorical',
            shuffle=shuffle,
            seed=42,
            interpolation='bilinear'
        )

    def create_generators(self) -> Tuple:
        """Create and return data generators for all sets.
        
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        try:
            self.logger.info("Creating data generators...")
            
            # Create training generator with augmentation
            train_gen = self._create_augmentation_pipeline(training=True)
            self.train_generator = self._create_dataset_generator(
                train_gen, 'train', shuffle=True
            )
            
            # Create validation generator
            val_gen = self._create_augmentation_pipeline(training=False)
            self.val_generator = self._create_dataset_generator(
                val_gen, 'val', shuffle=False
            )
            
            # Create test generator
            test_gen = self._create_augmentation_pipeline(training=False)
            self.test_generator = self._create_dataset_generator(
                test_gen, 'test', shuffle=False
            )
            
            self.logger.info("Data generators created successfully")
            
            return self.train_generator, self.val_generator, self.test_generator
            
        except Exception as e:
            self.logger.error(f"Error creating data generators: {str(e)}")
            raise

    def get_class_weights(self) -> Optional[dict]:
        """Calculate class weights for imbalanced datasets."""
        try:
            if not self.train_generator:
                return None
                
            total_samples = self.train_generator.samples
            num_classes = len(self.train_generator.class_indices)
            class_counts = [0] * num_classes
            
            for i in range(len(self.train_generator.classes)):
                class_counts[self.train_generator.classes[i]] += 1
            
            class_weights = {}
            for i in range(num_classes):
                weight = total_samples / (num_classes * class_counts[i])
                class_weights[i] = weight
                
            self.logger.info("Class weights calculated successfully")
            return class_weights
            
        except Exception as e:
            self.logger.error(f"Error calculating class weights: {str(e)}")
            return None

    def get_steps_per_epoch(self) -> int:
        """Calculate steps per epoch for training."""
        if self.train_generator:
            return self.train_generator.samples // self.config['batch_size']
        return 0 