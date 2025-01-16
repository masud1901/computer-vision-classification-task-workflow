"""Transfer Learning Models for Plant Disease Classification."""

from tensorflow.keras.applications import (
    ResNet50V2,
    EfficientNetB0,
    DenseNet121,
    InceptionV3,
    MobileNetV2
)
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class TransferLearningModels:
    """Collection of Transfer Learning Models for Classification."""
    
    def __init__(self, input_shape: tuple, num_classes: int):
        """Initialize transfer learning models.
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of classification classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = 1e-4

    def _add_classification_head(self, base_model):
        """Add classification head to the base model.
        
        Args:
            base_model: Pre-trained base model
            
        Returns:
            Model: Model with classification head
        """
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(
            512,
            activation='relu',
            kernel_regularizer=l2(self.weight_decay)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(
            256,
            activation='relu',
            kernel_regularizer=l2(self.weight_decay)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(
            self.num_classes,
            activation='softmax'
        )(x)
        
        return Model(inputs=base_model.input, outputs=predictions)

    def build_resnet50v2(self, trainable_layers: int = 50):
        """Build ResNet50V2 model.
        
        Args:
            trainable_layers (int): Number of trainable layers from top
            
        Returns:
            Model: ResNet50V2 model
        """
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
        return self._add_classification_head(base_model)

    def build_efficientnet(self, trainable_layers: int = 30):
        """Build EfficientNetB0 model.
        
        Args:
            trainable_layers (int): Number of trainable layers from top
            
        Returns:
            Model: EfficientNetB0 model
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
        return self._add_classification_head(base_model)

    def build_densenet(self, trainable_layers: int = 40):
        """Build DenseNet121 model.
        
        Args:
            trainable_layers (int): Number of trainable layers from top
            
        Returns:
            Model: DenseNet121 model
        """
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
        return self._add_classification_head(base_model)

    def build_inception(self, trainable_layers: int = 50):
        """Build InceptionV3 model.
        
        Args:
            trainable_layers (int): Number of trainable layers from top
            
        Returns:
            Model: InceptionV3 model
        """
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
        return self._add_classification_head(base_model)

    def build_mobilenet(self, trainable_layers: int = 30):
        """Build MobileNetV2 model.
        
        Args:
            trainable_layers (int): Number of trainable layers from top
            
        Returns:
            Model: MobileNetV2 model
        """
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
        return self._add_classification_head(base_model)


# Example usage:
if __name__ == "__main__":
    # Initialize transfer learning models
    transfer_models = TransferLearningModels(
        input_shape=(224, 224, 3),
        num_classes=15
    )
    
    # Build models
    resnet_model = transfer_models.build_resnet50v2()
    efficientnet_model = transfer_models.build_efficientnet()
    densenet_model = transfer_models.build_densenet()
    inception_model = transfer_models.build_inception()
    mobilenet_model = transfer_models.build_mobilenet()
    
    # Print model summaries
    print("ResNet50V2 Summary:")
    resnet_model.summary()
