"""Configuration settings for the plant disease classifier"""

CONFIG = {
    'input_folder': 'PlantVillage',
    'output_folder': 'Dataset',
    'image_size': (128, 128),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1
} 