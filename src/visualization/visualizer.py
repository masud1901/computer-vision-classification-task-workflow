"""Visualization utilities for training metrics and results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from matplotlib import rcParams


class Visualizer:
    """Visualization class for model metrics and performance."""

    def __init__(self):
        self.class_names = [
            'Pepper_bell_Bacterial_spot',
            'Pepper_bell_healthy',
            'Potato_Early_blight',
            'Potato_Late_blight',
            'Potato_healthy',
            'Tomato_Bacterial_spot',
            'Tomato_Early_blight',
            'Tomato_Late_blight',
            'Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite',
            'Tomato_Target_Spot',
            'Tomato_YellowLeaf_Curl_Virus',
            'Tomato_mosaic_virus',
            'Tomato_healthy'
        ]
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Setup global plotting style."""
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 14
        plt.style.use('seaborn')

    def plot_training_history(self, history):
        """Plot training history metrics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='train', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=16)
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='train', linewidth=2)
        ax2.plot(history.history['val_loss'], label='validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=16)
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Loss', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, class_indices):
        """Plot confusion matrix."""
        plt.figure(figsize=(15, 12))
        cm = confusion_matrix(y_true, y_pred)
        
        # Create a mapping from index to class name
        idx_to_class = {v: k for k, v in class_indices.items()}
        labels = [idx_to_class[i] for i in range(len(class_indices))]
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_roc_curves(self, test_generator, model, true_labels, pred_probs):
        """Plot ROC curves for all classes."""
        # Binarize the true labels for ROC
        num_classes = len(self.class_names)
        true_labels_binarized = label_binarize(true_labels, 
                                             classes=list(range(num_classes)))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i],
                                         pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(15, 10))
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                       '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'])

        for i, (color, class_name) in enumerate(zip(colors, self.class_names)):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f'ROC curve of {class_name} (area = {roc_auc[i]:0.2f})'
            )

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.7, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        return roc_auc

    def plot_metrics_summary(self, metrics):
        """Plot summary of various metrics."""
        plt.figure(figsize=(12, 6))
        metrics_names = list(metrics.keys())
        values = list(metrics.values())

        plt.bar(metrics_names, values)
        plt.title('Model Metrics Summary', fontsize=16)
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()