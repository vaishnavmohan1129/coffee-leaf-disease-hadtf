import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from torchvision import transforms
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------

MODEL_PATH = "hadtf_improved.pth"
CLASS_NAMES = ['Cercospora', 'Miner', 'Phoma', 'Rust']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# HADTF MODEL (EXACT SAME AS NOTEBOOK)
# ---------------------------

class HADTF(nn.Module):
    def __init__(self, num_classes=4):
        super(HADTF, self).__init__()

        # CNN Backbone (ResNet50)
        self.cnn = models.resnet50(pretrained=False)
        self.cnn.fc = nn.Identity()

        # ViT Backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = nn.Identity()

        self.cnn_dim = 2048
        self.vit_dim = 768

        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.cnn_dim + self.vit_dim, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)

        fused = torch.cat((
            self.alpha * cnn_features,
            (1 - self.alpha) * vit_features
        ), dim=1)

        return self.classifier(fused)


# ---------------------------
# LOAD MODEL
# ---------------------------

@st.cache_resource
def load_model():
    model = HADTF(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------------------
# IMAGE TRANSFORM
# ---------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.title("☕ Coffee Leaf Disease Detection (HADTF Model)")
st.write("Upload a coffee leaf image to detect the disease.")



uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_percent = confidence.item() * 100

    st.subheader(f"🩺 Predicted Disease: {predicted_class}")
    st.write(f"Confidence: {confidence_percent:.2f}%")

    st.write("### Class Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        st.write(f"{CLASS_NAMES[i]}: {prob.item()*100:.2f}%")