# ✋ Hand Recognition Using MediaPipe

This project uses [MediaPipe](https://mediapipe.dev/) to perform real-time hand detection and recognition. It identifies hand landmarks, tracks movement, and classifies gestures. The end goal is to use this model to build a **presentation control app**, allowing users to move between slides using simple hand gestures — no clicker needed!

---

## 🚀 Features

- Real-time hand tracking using MediaPipe  
- Detection of 21 hand landmarks  
- Custom gesture recognition using angles and distances  
- Easily extendable to support more gestures and actions  

---

## 📁 Project Files

- `playground.ipynb` – Main notebook for hand tracking and gesture classification  
- `requirements.yml` – Conda environment file to recreate the setup  

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hand-recognition-app.git
cd hand-recognition-app
```

### 2. Create and activate the environment

conda env create -f requirements.yml
conda activate hand-recognition

### 3. Have fun with either the notebook or the main.py file

    jupyter notebook playground.ipynb
    python main.py

## 🎯 Future Plans

    ✅ Build a presentation controller app to navigate slides using gestures

    ✅ Add a simple GUI for easier usage

    ⏳ Add more gestures (e.g., zoom, pointer) and custom action mapping

## 🖼️ Sample Output

![Hand Recognition Demo](misc/fun.gif)

## 🤝 Contributing

Pull requests are welcome! Feel free to fork this project and suggest improvements or additional features.

>Made with ❤️ using Python, OpenCV, and MediaPipe
