import os
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np

MODEL_PATH = "models/disease_resnet18_best.onnx"
IMAGE_FOLDER = "data_raw/leaf_spot"
class_names = ["dollar_spot", "brown_patch", "leaf_spot", "pythium"]

ort_session = ort.InferenceSession(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_FOLDER, filename)
        image = Image.open(path).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).numpy()  # (1, 3, 224, 224)

        outputs = ort_session.run(None, {"input": input_tensor})
        probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[0][predicted_idx] * 100

        print(f"{filename} → {predicted_class} ({confidence:.2f}%)")