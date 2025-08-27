# BD Freshwater Fish Detection and Classification

This project classifies freshwater fish species using a **Vision Transformer (ViT)** model and provides a **Streamlit app** for real-time predictions and information on them.

---

## Features
- Trains a **ViT model** for fish species classification.
- Interactive **Streamlit app** for uploading images and predicting fish species.
- Supports  12 bd freshwater fish species.

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- Plotly
- Requests
- BeautifulSoup4

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd fish_detection_app
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Pre-trained Model
Download the pre-trained Vision Transformer model from Google Drive: https://drive.google.com/file/d/1lBkY_JXJc6Bj2ovZVxolcnppYETMCAsb/view?usp=drive_link

Download Model

Place the downloaded model in the model/ folder before running the app.

Requirements (requirements.txt)
text
Copy code
streamlit==1.28.0
numpy
opencv-python
plotly
requests
beautifulsoup4
tensorflow
keras
Usage
Run the Streamlit app:

bash
Copy code
streamlit run main_app.py
Upload a fish image to get the predicted species in real-time.
And the website to get information on this species

Folder Structure
graphql
Copy code
fish_detection_app/
├─ main_app.py          # Streamlit app
├─ model/               # Trained ViT model and weights
├─ data/                # Sample images or datasets
├─ requirements.txt
└─ README.md
License
MIT License


