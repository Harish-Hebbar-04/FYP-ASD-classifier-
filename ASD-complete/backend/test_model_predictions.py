import torch
import numpy as np
from model import load_trained_model

# Load the trained model
print("Loading trained model...")
model_info = load_trained_model("vit_asd_best.pth")
model = model_info.model
device = model_info.device
print(f"Model loaded successfully!")
print(f"Model architecture: {model_info.architecture}")
print(f"Class names: {model_info.class_names}")
print(f"Input size: {model_info.input_size}")
print(f"Device: {device}")
print()

# Create 3 different random test inputs
print("Creating 3 different random test inputs...")
test_inputs = []
for i in range(3):
    # Vision Transformer expects (batch, channels, height, width)
    random_input = torch.randn(1, 3, 224, 224)
    test_inputs.append(random_input)
    print(f"Test input {i+1} shape: {random_input.shape}")

print("\n" + "="*60)
print("Running predictions on each test input...")
print("="*60)

# Run predictions on each
predictions = []
model.eval()
with torch.no_grad():
    for i, test_input in enumerate(test_inputs):
        test_input = test_input.to(device)
        output = model(test_input)
        predictions.append(output)
        print(f"\nTest input {i+1} prediction:")
        print(f"  Output shape: {output.shape}")
        print(f"  Raw logits: {output}")
        probs = torch.softmax(output, dim=1)
        print(f"  Probabilities: {probs}")
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = model_info.class_names[predicted_class]
        print(f"  Predicted class: {predicted_class} ({predicted_label})")

print("\n" + "="*60)
print("Comparison Summary:")
print("="*60)
all_same = torch.allclose(predictions[0], predictions[1]) and torch.allclose(predictions[1], predictions[2])
print(f"Are all predictions the same? {all_same}")
if not all_same:
    print(f"Prediction 1: {predictions[0]}")
    print(f"Prediction 2: {predictions[1]}")
    print(f"Prediction 3: {predictions[2]}")
