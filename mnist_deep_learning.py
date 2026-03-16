import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)

class MNISTDeepLearning:
    def __init__(self):
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_custom = None
        
    def load_data(self, use_custom_csv=True, csv_path='mnist_dataset.csv'):
        print("Loading MNIST data...")
        
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        if use_custom_csv and os.path.exists(csv_path):
            print(f"\nLoading custom data from {csv_path}...")
            try:
                chunk_size = 1000
                chunks = []
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    chunks.append(chunk)
                custom_data = pd.concat(chunks, ignore_index=True)
                
                self.X_custom = custom_data.values.reshape(-1, 28, 28)
                print(f"Custom data shape: {self.X_custom.shape}")
                
            except Exception as e:
                print(f"Error loading custom CSV: {e}")
                print("Continuing with built-in MNIST data only.")
                self.X_custom = None
        
        print("Data loading completed!\n")
    
    def preprocess_data(self):
        print("Preprocessing data...")
        
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        if self.X_custom is not None:
            self.X_custom = self.X_custom.astype('float32') / 255.0
        
        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)
        
        print("Preprocessing completed!")
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        
        if self.X_custom is not None:
            print(f"Custom data shape: {self.X_custom.shape}")
        print()
    
    def create_model(self):
        print("Creating deep neural network model...")
        
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("\nModel Architecture:")
        print("=" * 50)
        self.model.summary()
        print("=" * 50)
        print()
    
    def train_model(self, epochs=20, batch_size=128, validation_split=0.1):
        print(f"Training model for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Validation split: {validation_split}")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
    
    def evaluate_model(self):
        print("Evaluating model performance...")
        
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        print("\nClassification Report:")
        print("=" * 50)
        print(classification_report(y_true_classes, y_pred_classes))
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return test_accuracy, test_loss, cm, y_pred_classes, y_true_classes
    
    def predict_custom_data(self):
        if self.X_custom is not None:
            print("Making predictions on custom data...")
            predictions = self.model.predict(self.X_custom, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            print(f"Custom data predictions shape: {predictions.shape}")
            print(f"Sample predictions (first 20): {predicted_classes[:20]}")
            
            return predicted_classes
        else:
            print("No custom data available for prediction.")
            return None
    
    def plot_training_history(self):
        if self.history is None:
            print("No training history available.")
            return
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if 'lr' in self.history.history:
            plt.subplot(1, 3, 3)
            plt.plot(self.history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(self, y_pred_classes, y_true_classes, num_samples=20):
        plt.figure(figsize=(20, 8))
        
        for i in range(num_samples):
            plt.subplot(4, 5, i + 1)
            plt.imshow(self.X_test[i], cmap='gray')
            
            true_label = y_true_classes[i]
            pred_label = y_pred_classes[i]
            
            if true_label == pred_label:
                color = 'green'
                title = f'✓ True: {true_label}, Pred: {pred_label}'
            else:
                color = 'red'
                title = f'✗ True: {true_label}, Pred: {pred_label}'
            
            plt.title(title, color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='mnist_deep_model.h5'):
        if self.model is not None:
            self.model.save(filename)
            print(f"Model saved as {filename}")
        else:
            print("No model to save.")
    
    def load_model(self, filename='mnist_deep_model.h5'):
        if os.path.exists(filename):
            self.model = keras.models.load_model(filename)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} not found.")


def main():
    print("MNIST Image Classification using Deep Neural Networks")
    print("=" * 60)
    
    classifier = MNISTDeepLearning()
    
    classifier.load_data(use_custom_csv=True, csv_path='mnist_dataset.csv')
    classifier.preprocess_data()
    
    classifier.create_model()
    classifier.train_model(epochs=15, batch_size=128, validation_split=0.1)
    
    test_accuracy, test_loss, cm, y_pred_classes, y_true_classes = classifier.evaluate_model()
    
    custom_predictions = classifier.predict_custom_data()
    
    print("\nCreating visualizations...")
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(cm)
    classifier.plot_sample_predictions(y_pred_classes, y_true_classes)
    
    classifier.save_model('mnist_deep_model.h5')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    if custom_predictions is not None:
        print(f"Custom data predictions completed for {len(custom_predictions)} samples")
    
    print("\nFiles generated:")
    print("- mnist_deep_model.h5 (saved model)")
    print("- training_history.png (training curves)")
    print("- confusion_matrix.png (confusion matrix)")
    print("- sample_predictions.png (sample predictions)")
    print("\nClassification completed!")


if __name__ == "__main__":
    main()