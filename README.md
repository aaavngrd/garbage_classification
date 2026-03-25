# Garbage Classification System

##  Project Overview
This project is a machine learning–based system for automatic classification of waste images.
The system analyzes an input image and predicts the type of garbage ,
helping to improve waste sorting and recycling processes.

The project was developed as part of an academic assignment and includes:
- model training using PyTorch
- REST API for inference using FastAPI

---

## Technologies Used
- Python 3
- PyTorch
- FastAPI
- torchvision

---

## Dataset
The project uses an image dataset of household waste with the following classes:
- paper
- plastic
- glass
- metal
- organic

The dataset is split into training, validation, and test subsets.

**Note:**  
Due to size limitations, the dataset is not included in the repository.
Dataset directories (`data/`, `raw_dataset/`, `inference_images/`) are excluded via `.gitignore`.

Expected local dataset structure:
data/
├── train/
├── val/
└── test/

---

## Project Structure

garbage_classification/
├──data/
│   ├── train/
│   ├── val/
│   └── test/
├── server.py # FastAPI server for inference
├── inference.py # CLI inference script
├── train.py # Model training script
├── split_dataset.py # Dataset splitting utility
├── .gitignore
├── README.md