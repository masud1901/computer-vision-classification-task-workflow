"""Ablation Study for CNN Model Architecture."""

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Dense, Dropout,
    GlobalAveragePooling2D, LeakyReLU
)
from src.models.cnn_model import (
    ChannelAttention,
    SpatialAttention,
    PlantDiseaseModel
)


class AblationStudy:
    """Ablation study for analyzing model components."""
    
    def __init__(self, input_shape: tuple, num_classes: int):
        """Initialize ablation study.
        
        Args:
            input_shape: Input image dimensions
            num_classes: Number of classification classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_reg = regularizers.l2(1e-4)

    def baseline_model(self):
        """Basic CNN model without any additional components."""
        inputs = Input(shape=self.input_shape)
        
        x = Conv2D(64, 3, padding='same', kernel_regularizer=self.l2_reg)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        x = Conv2D(128, 3, padding='same', kernel_regularizer=self.l2_reg)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=self.l2_reg)(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs, name='baseline')

    def attention_only_model(self):
        """Model with only attention mechanisms."""
        inputs = Input(shape=self.input_shape)
        
        x = Conv2D(64, 3, padding='same', kernel_regularizer=self.l2_reg)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        # Add attention modules
        x = ChannelAttention()(x)
        x = SpatialAttention()(x)
        
        x = Conv2D(128, 3, padding='same', kernel_regularizer=self.l2_reg)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=self.l2_reg)(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs, name='attention_only')

    def dense_only_model(self):
        """Model with only dense connections."""
        inputs = Input(shape=self.input_shape)
        
        # First dense block
        x = self._dense_block(inputs, num_layers=4, growth_rate=32)
        x = Conv2D(128, 1, padding='same', kernel_regularizer=self.l2_reg)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=self.l2_reg)(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs, name='dense_only')

    def inception_only_model(self):
        """Model with only inception modules."""
        inputs = Input(shape=self.input_shape)
        
        # First inception module
        x = self._inception_module(inputs, filters=32)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        # Second inception module
        x = self._inception_module(x, filters=64)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=self.l2_reg)(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs, name='inception_only')

    def run_ablation_study(self):
        """Run complete ablation study with all model variants."""
        models = {
            'baseline': self.baseline_model(),
            'attention_only': self.attention_only_model(),
            'dense_only': self.dense_only_model(),
            'inception_only': self.inception_only_model(),
            'full_model': PlantDiseaseModel(
                self.input_shape,
                self.num_classes
            ).build_model()
        }
        return models


def compare_models(models_dict):
    """Compare model architectures and parameters."""
    results = {}
    for name, model in models_dict.items():
        trainable_params = sum(
            [tf.keras.backend.count_params(w) for w in model.trainable_weights]
        )
        non_trainable_params = sum(
            [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
        )
        results[name] = {
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params': trainable_params + non_trainable_params
        }
    return results


# Example usage
if __name__ == "__main__":
    # Initialize ablation study
    ablation = AblationStudy(
        input_shape=(224, 224, 3),
        num_classes=15
    )
    
    # Get all model variants
    models = ablation.run_ablation_study()
    
    # Compare model architectures
    comparison = compare_models(models)
    
    # Print results
    for model_name, params in comparison.items():
        print(f"\n{model_name} Architecture:")
        print(f"Trainable params: {params['trainable_params']:,}")
        print(f"Non-trainable params: {params['non_trainable_params']:,}")
        print(f"Total params: {params['total_params']:,}")
        print("-" * 50) 