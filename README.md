# 🎬 Movie-Recommendation-System

## Overview
This project implements a **User-Based Collaborative Filtering (UBCF)** Recommendation System using **Apache Spark (PySpark)**. It predicts movie ratings for users and recommends top movies based on **user–user similarity** computed via the **Pearson correlation coefficient**.

---

## Features
- Developed using **Apache Spark** for distributed, large-scale data processing.  
- Implements **User–User Collaborative Filtering**.  
- Computes similarity using **Pearson correlation**.  
- Predicts unseen movie ratings using a **weighted similarity formula**.  
- Displays **Top-N recommended movies** for a chosen user.  
- Provides detailed breakdown of prediction and contributing users.

---

## 📂 Dataset
This project uses the public **MovieLens dataset** available at:

> https://github.com/sankalpjain99/Movie-recommendation-system.git

### Dataset Files:
- **`ratings.csv`** — contains user–movie ratings  
- **`movies.csv`** — contains movie metadata  



---

## 📸 Screenshots

### 1. Top Recommendation for User 10
![Top Recommendation for User 10](https://github.com/adarshms444/User-Based-Collaborative-Movie-Recommendation-System/blob/main/images/Movienite1.png)

### 2. Top Recommendation for User 100
![Top Recommendation for User 100](https://github.com/adarshms444/User-Based-Collaborative-Movie-Recommendation-System/blob/main/images/Movienite3.png)

### 3. Deeper Dive Analytics of User 100
![Deeper Analytics for User 100](https://github.com/adarshms444/User-Based-Collaborative-Movie-Recommendation-System/blob/main/images/Movienite4.png)


---

## ⚙️ Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/<your-username>/UserBasedCF-Spark.git
cd UserBasedCF-Spark
```



