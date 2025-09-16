# BD Freshwater Fish Detection and Classification

This project classifies freshwater fish species using a **MobileNetV2** model and provides a **Streamlit app** for real-time predictions and information on them.
Image source: [https://data.mendeley.com/datasets/2gkj4h388d/3](https://data.mendeley.com/datasets/2gkj4h388d/3)

---

## Features

* Trains a **MobileNetV2 model** for fish species classification.
* Interactive **Streamlit app** for uploading images and predicting fish species.
* Supports **12 Bangladeshi freshwater fish species**.

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* Plotly
* Requests
* BeautifulSoup4

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/marufmullah50/Fish-detection-app.git
cd fish_detection_app
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Pre-trained Model
   Download the pre-trained **MobileNetV2 model** from Google Drive:
   [Download Model](https://drive.google.com/file/d/1lBkY_JXJc6Bj2ovZVxolcnppYETMCAsb/view?usp=drive_link)

Place the downloaded model in the `model/` folder before running the app.

---

## Requirements (`requirements.txt`)

```
streamlit==1.28.0
numpy
opencv-python
plotly
requests
beautifulsoup4
tensorflow
keras
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run main_app.py
```

* Upload a fish image to get the predicted species in real-time.
* The app also fetches additional information about the predicted species from the web.

---

## Folder Structure

```
fish_detection_app/
├─ main_app.py          # Streamlit app
├─ model/               # Trained MobileNetV2 model and weights                
├─ requirements.txt
```

---

## License

MIT License
