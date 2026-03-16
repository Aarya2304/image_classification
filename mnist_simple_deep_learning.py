import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class MNISTSimpleDeepLearning:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_custom = None
        
    def load_mnist_data(self):
        print("Loading MNIST data...")
        
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data, mnist.target.astype(int)
            
            subset_size = 10000
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = y[indices]
            
            print(f"Data shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Unique labels: {np.unique(y)}")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading from sklearn: {e}")
            print("Generating synthetic MNIST-like data...")
            X = np.random.randint(0, 256, (1000, 784))
            y = np.random.randint(0, 10, 1000)
            return X, y
    
    def load_custom_data(self, csv_path='mnist_dataset.csv'):
        if os.path.exists(csv_path):
            print(f"Loading custom data from {csv_path}...")
            try:
                chunk_size = 1000
                chunks = []
                chunk_count = 0
                max_chunks = 5
                
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    chunk_count += 1
                    if chunk_count >= max_chunks:
                        break
                
                custom_data = pd.concat(chunks, ignore_index=True)
                self.X_custom = custom_data.values
                print(f"Custom data shape: {self.X_custom.shape}")
                
                return self.X_custom
                
            except Exception as e:
                print(f"Error loading custom CSV: {e}")
                return None
        else:
            print(f"Custom data file {csv_path} not found.")
            return None
    
    def preprocess_data(self, X, y):
        print("Preprocessing data...")
        
        X = X.astype('float32') / 255.0
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print("Preprocessing completed!")
        
        return X_train, X_test, y_train, y_test
    
    def create_and_train_model(self, max_iter=100):
        print("Creating and training deep neural network...")
        print("Architecture: 784 -> 512 -> 256 -> 128 -> 64 -> 10")
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=128,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=max_iter,
            random_state=42,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        print("Training started...")
        self.model.fit(self.X_train, self.y_train)
        print("Training completed!")
        
        return self.model
    
    def evaluate_model(self):
        print("\nEvaluating model performance...")
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nClassification Report:")
        print("=" * 50)
        print(classification_report(self.y_test, y_pred))
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        return accuracy, cm, y_pred
    
    def predict_custom_data(self):
        if self.X_custom is not None and self.model is not None:
            print("Making predictions on custom data...")
            
            X_custom_norm = self.X_custom.astype('float32') / 255.0
            
            predictions = self.model.predict(X_custom_norm)
            
            print(f"Custom data predictions shape: {predictions.shape}")
            print(f"Sample predictions (first 20): {predictions[:20]}")
            
            return predictions
        else:
            print("No custom data or model available for prediction.")
            return None
    
    def plot_training_history(self):
        if hasattr(self.model, 'loss_curve_'):
            plt.figure(figsize=(10, 6))
            plt.plot(self.model.loss_curve_)
            plt.title('Training Loss Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No training history available.")
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(self, y_pred, num_samples=20):
        plt.figure(figsize=(20, 8))
        
        for i in range(min(num_samples, len(self.X_test))):
            plt.subplot(4, 5, i + 1)
            
            image = self.X_test[i].reshape(28, 28)
            plt.imshow(image, cmap='gray')
            
            true_label = self.y_test[i]
            pred_label = y_pred[i]
            
            if true_label == pred_label:
                color = 'green'
                title = f'✓ True: {true_label}, Pred: {pred_label}'
            else:
                color = 'red'
                title = f'✗ True: {true_label}, Pred: {pred_label}'
            
            plt.title(title, color=color, fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, accuracy, predictions):
        results = {
            'accuracy': accuracy,
            'model_info': str(self.model),
            'predictions_sample': predictions[:100].tolist() if predictions is not None else []
        }
        
        with open('results.txt', 'w') as f:
            f.write(f"MNIST Deep Learning Classification Results\n")
            f.write(f"==========================================\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write(f"Model Configuration:\n{self.model}\n\n")
            
            if predictions is not None:
                f.write(f"Sample Custom Predictions: {predictions[:20]}\n")
        
        print("Results saved to results.txt")


def main():
    print("MNIST Image Classification using Deep Neural Networks")
    print("=" * 60)
    
    classifier = MNISTSimpleDeepLearning()
    
    X, y = classifier.load_mnist_data()
    X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)
    
    custom_data = classifier.load_custom_data('mnist_dataset.csv')
    
    model = classifier.create_and_train_model(max_iter=50)
    
    accuracy, cm, y_pred = classifier.evaluate_model()
    
    custom_predictions = classifier.predict_custom_data()
    
    print("\nCreating visualizations...")
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(cm)
    classifier.plot_sample_predictions(y_pred)
    
    classifier.save_results(accuracy, custom_predictions)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if custom_predictions is not None:
        print(f"Custom data predictions completed for {len(custom_predictions)} samples")
    
    print("\nFiles generated:")
    print("- training_loss.png (training curve)")
    print("- confusion_matrix.png (confusion matrix)")
    print("- sample_predictions.png (sample predictions)")
    print("- results.txt (detailed results)")
    print("\nClassification completed!")


if __name__ == "__main__":
    main()