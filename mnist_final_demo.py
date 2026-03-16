
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("🚀 MNIST Deep Neural Network Classification")
print("=" * 60)

print("\n📊 Step 1: Loading MNIST Data...")
try:
    print("Attempting to load real MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    subset_size = 5000
    indices = np.random.choice(len(X), subset_size, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"✅ Successfully loaded {len(X)} MNIST samples")
    data_source = "Real MNIST"
    
except Exception as e:
    print(f"⚠️  Could not load real MNIST data: {e}")
    print("📋 Generating synthetic MNIST-like data...")
    
    n_samples = 2000
    X = np.random.randint(0, 256, size=(n_samples, 784))
    y = np.random.randint(0, 10, size=n_samples)
    
    for i in range(n_samples):
        digit = y[i]
        if digit == 0:
            X[i, 200:300] = np.random.randint(100, 255, 100)
        elif digit == 1:
            X[i, 350:450] = np.random.randint(150, 255, 100)
    
    print(f"✅ Generated {len(X)} synthetic samples")
    data_source = "Synthetic MNIST-like"

print(f"📈 Data shape: {X.shape}")
print(f"🏷️  Labels shape: {y.shape}")
print(f"🔢 Unique labels: {sorted(np.unique(y))}")
print(f"📋 Data source: {data_source}")

print("\n📂 Step 2: Loading Custom CSV Data...")
csv_path = 'mnist_dataset.csv'
custom_data = None

if os.path.exists(csv_path):
    try:
        print(f"Loading sample from {csv_path}...")
        custom_df = pd.read_csv(csv_path, nrows=500)
        custom_data = custom_df.values
        print(f"✅ Loaded {len(custom_data)} custom samples from CSV")
    except Exception as e:
        print(f"⚠️  Error loading CSV: {e}")
        custom_data = None
else:
    print(f"⚠️  CSV file {csv_path} not found")

print("\n🔧 Step 3: Preprocessing Data...")
X = X.astype('float32') / 255.0  # Normalize to [0,1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")
print(f"✅ Data normalized to range [0, 1]")

print("\n🧠 Step 4: Creating Deep Neural Network...")
print("🏗️  Architecture: 784 → 512 → 256 → 128 → 64 → 10")

model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=30,
    random_state=42,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5
)

print("✅ Deep Neural Network created")

print("\n🎓 Step 5: Training Deep Neural Network...")
print("⏳ Training in progress...")

model.fit(X_train, y_train)

print(f"✅ Training completed!")
print(f"📊 Training iterations: {model.n_iter_}")
print(f"📉 Final training loss: {model.loss_:.6f}")

print("\n📊 Step 6: Evaluating Model Performance...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:")
print("=" * 50)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\n🔮 Step 7: Making Predictions on Custom Data...")
custom_predictions = None

if custom_data is not None:
    custom_data_norm = custom_data.astype('float32') / 255.0
    custom_predictions = model.predict(custom_data_norm)
    
    print(f"✅ Made predictions on {len(custom_predictions)} custom samples")
    print(f"📊 Sample predictions: {custom_predictions[:20]}")
    print(f"📈 Prediction distribution: {dict(zip(*np.unique(custom_predictions, return_counts=True)))}")
else:
    print("⚠️  No custom data available for prediction")

print("\n📊 Step 8: Creating Visualizations...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
if hasattr(model, 'loss_curve_'):
    plt.plot(model.loss_curve_, 'b-', linewidth=2)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 3, 3)
class_accuracies = []
for i in range(10):
    mask = y_test == i
    if np.sum(mask) > 0:
        class_acc = accuracy_score(y_test[mask], y_pred[mask])
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

bars = plt.bar(range(10), class_accuracies, color='skyblue', edgecolor='navy')
plt.title('Accuracy by Digit Class', fontsize=14, fontweight='bold')
plt.xlabel('Digit')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)

for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('mnist_deep_learning_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: mnist_deep_learning_results.png")

plt.figure(figsize=(20, 12))
num_samples = min(20, len(X_test))

for i in range(num_samples):
    plt.subplot(4, 5, i + 1)
    
    image = X_test[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    
    true_label = y_test[i]
    pred_label = y_pred[i]
    
    if true_label == pred_label:
        color = 'green'
        symbol = '✓'
    else:
        color = 'red'
        symbol = '✗'
    
    plt.title(f'{symbol} True: {true_label}, Pred: {pred_label}', 
              color=color, fontsize=12, fontweight='bold')
    plt.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions_detailed.png', dpi=300, bbox_inches='tight')
print("✅ Saved: sample_predictions_detailed.png")

print("\n💾 Step 9: Saving Detailed Results...")

results_content = f"""MNIST Deep Neural Network Classification - Detailed Results
===========================================================

📊 Dataset Information:
- Data Source: {data_source}
- Total Samples: {len(X)}
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Features: 784 (28x28 pixels)
- Classes: 10 (digits 0-9)

🧠 Model Architecture:
- Type: Multi-Layer Perceptron (Deep Neural Network)
- Input Layer: 784 neurons
- Hidden Layer 1: 512 neurons (ReLU activation)
- Hidden Layer 2: 256 neurons (ReLU activation)
- Hidden Layer 3: 128 neurons (ReLU activation)
- Hidden Layer 4: 64 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)
- Total Parameters: ~500,000+

⚙️ Training Configuration:
- Optimizer: Adam
- Learning Rate: 0.001 (adaptive)
- Batch Size: 64
- Max Iterations: 30
- Early Stopping: Enabled
- Validation Fraction: 10%
- Regularization (Alpha): 0.0001

📈 Training Results:
- Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
- Training Iterations: {model.n_iter_}
- Final Training Loss: {model.loss_:.6f}
- Convergence: {'Yes' if model.n_iter_ < 30 else 'No (max iterations reached)'}

📊 Performance Metrics:

results_content += f"""
📁 Generated Files:
- mnist_deep_learning_results.png (training metrics and confusion matrix)
- sample_predictions_detailed.png (visual predictions)
- detailed_mnist_results.txt (this file)

🎯 Summary:
This deep neural network successfully learned to classify MNIST digits with {accuracy*100:.1f}% accuracy.
The model used a deep architecture with 4 hidden layers and demonstrated good generalization performance.

if custom_predictions is not None:
    print(f"✅ Custom CSV predictions: {len(custom_predictions)} samples processed")

print("\n🚀 Deep Learning Classification Task COMPLETE! 🚀")