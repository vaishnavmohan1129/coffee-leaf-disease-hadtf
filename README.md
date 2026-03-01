# Coffee Leaf Disease Detection using HADTF
This project is a deep learning web application that detects diseases in coffee leaves using a Hybrid Attention-based Dual Transformer Framework (HADTF) model.

The model combines:
-> ResNet50 (CNN backbone)
-> Vision Transformer (ViT)
-> Feature fusion with learnable weighting

The application is built using Streamlit and deployed from GitHub.

  Features
Upload coffee leaf images

Detect 4 major coffee leaf diseases:
.Cercospora
.Miner
.Phoma
.Rust

Displays prediction confidence

Shows probability for all classes

  --Model Architecture (HADTF)
The HADTF model combines:

.ResNet50 for spatial feature extraction
.Vision Transformer (ViT-B/16) for global attention
.eature fusion using a learnable parameter
.Dropout regularization
.Fully connected classification layer

Model file:

hadtf_improved.pth
  Project Structure
coffee-leaf-disease-hadtf/
│
├── app.py
├── hadtf_improved.pth
├── requirements.txt
├── README.md
└── .gitignore


  Installation (Run Locally)
1️ Clone Repository

git clone https://github.com/vaishnavmohan1129/coffee-leaf-disease-hadtf.git
cd coffee-leaf-disease-hadtf

2️ Create Virtual Environment

python -m venv venv
venv\Scripts\activate   # Windows

3️ Install Dependencies

pip install -r requirements.txt

4️ Run Application

streamlit run app.py
