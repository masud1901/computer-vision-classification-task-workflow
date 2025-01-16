"""Explainable AI Analysis for Plant Disease Classification."""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from xplique.attributions import (
    GradCAM, Saliency, GradientInput, IntegratedGradients,
    SmoothGrad, Occlusion, Rise, GuidedBackprop
)
from lime import lime_image
import shap
from typing import List, Tuple, Dict
import logging


class XAIAnalyzer:
    """XAI Analysis for Plant Disease Classification Models."""
    
    def __init__(self, model_path: str, class_names: List[str]):
        """Initialize XAI analyzer.
        
        Args:
            model_path: Path to saved model
            class_names: List of class names
        """
        self.model = load_model(model_path)
        self.class_names = class_names
        self.logger = self._setup_logger()
        
        # Create output directories
        os.makedirs('explanations', exist_ok=True)
        os.makedirs('explanations/gradcam', exist_ok=True)
        os.makedirs('explanations/lime', exist_ok=True)
        os.makedirs('explanations/shap', exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for XAI analysis."""
        logger = logging.getLogger('XAIAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        return image

    def generate_gradcam(
        self,
        image: np.ndarray,
        layer_name: str = None
    ) -> Tuple[np.ndarray, Dict]:
        """Generate GradCAM visualization.
        
        Args:
            image: Input image
            layer_name: Target layer name for GradCAM
            
        Returns:
            Tuple of heatmap and metrics
        """
        self.logger.info("Generating GradCAM explanation...")
        
        # Initialize GradCAM
        gradcam = GradCAM(self.model)
        
        # Generate explanation
        preprocessed_image = self.preprocess_image(image)
        explanation = gradcam(
            preprocessed_image,
            class_index=None  # Uses predicted class
        )
        
        return explanation[0], {
            'method': 'GradCAM',
            'layer': layer_name
        }

    def generate_lime(
        self,
        image: np.ndarray,
        num_samples: int = 1000
    ) -> Tuple[np.ndarray, Dict]:
        """Generate LIME explanation.
        
        Args:
            image: Input image
            num_samples: Number of perturbation samples
            
        Returns:
            Tuple of explanation and metrics
        """
        self.logger.info("Generating LIME explanation...")
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image,
            self.model.predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get visualization
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        
        return mask, {
            'method': 'LIME',
            'num_samples': num_samples
        }

    def generate_shap(
        self,
        image: np.ndarray,
        background_images: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Generate SHAP values.
        
        Args:
            image: Input image
            background_images: Background images for SHAP
            
        Returns:
            Tuple of SHAP values and metrics
        """
        self.logger.info("Generating SHAP explanation...")
        
        # Initialize explainer
        explainer = shap.GradientExplainer(
            (self.model.input, self.model.output),
            background_images
        )
        
        # Generate SHAP values
        shap_values = explainer.shap_values(
            self.preprocess_image(image)
        )
        
        return shap_values[0], {
            'method': 'SHAP',
            'background_size': len(background_images)
        }

    def plot_explanation(
        self,
        image: np.ndarray,
        explanation: np.ndarray,
        method: str,
        metrics: Dict,
        save_path: str
    ) -> None:
        """Plot and save explanation visualization.
        
        Args:
            image: Original image
            explanation: Generated explanation
            method: Explanation method name
            metrics: Explanation metrics
            save_path: Path to save visualization
        """
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot explanation
        plt.subplot(1, 2, 2)
        if method == 'GradCAM':
            plt.imshow(image)
            plt.imshow(explanation, cmap='jet', alpha=0.5)
        elif method == 'LIME':
            plt.imshow(explanation, cmap='RdBu', alpha=0.5)
        elif method == 'SHAP':
            shap.image_plot(explanation, image, show=False)
        
        plt.title(f'{method} Explanation')
        plt.axis('off')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_image(
        self,
        image: np.ndarray,
        background_images: np.ndarray = None
    ) -> Dict:
        """Run complete XAI analysis on an image.
        
        Args:
            image: Input image
            background_images: Background images for SHAP
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        # Generate GradCAM explanation
        gradcam_exp, gradcam_metrics = self.generate_gradcam(image)
        results['gradcam'] = {
            'explanation': gradcam_exp,
            'metrics': gradcam_metrics
        }
        
        # Generate LIME explanation
        lime_exp, lime_metrics = self.generate_lime(image)
        results['lime'] = {
            'explanation': lime_exp,
            'metrics': lime_metrics
        }
        
        # Generate SHAP explanation if background images provided
        if background_images is not None:
            shap_exp, shap_metrics = self.generate_shap(
                image,
                background_images
            )
            results['shap'] = {
                'explanation': shap_exp,
                'metrics': shap_metrics
            }
        
        # Plot and save explanations
        for method, result in results.items():
            save_path = f'explanations/{method}/{method}_explanation.png'
            self.plot_explanation(
                image,
                result['explanation'],
                method.upper(),
                result['metrics'],
                save_path
            )
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize XAI analyzer
    analyzer = XAIAnalyzer(
        model_path='models/final_model.h5',
        class_names=[
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
    )
    
    # Load and analyze sample image
    # Replace with your image loading logic
    sample_image = np.random.rand(224, 224, 3)  # Example image
    background_images = np.random.rand(10, 224, 224, 3)  # Example background
    
    # Run analysis
    results = analyzer.analyze_image(sample_image, background_images)
    
    # Print results summary
    for method, result in results.items():
        print(f"\n{method.upper()} Analysis:")
        for metric, value in result['metrics'].items():
            print(f"{metric}: {value}") 