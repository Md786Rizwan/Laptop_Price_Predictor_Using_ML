💻 Laptop Price Predictor

## How to run locally

pip install -r requirements.txt  
streamlit run app.py


![alt text](image.png)

🚀 Overview

The Laptop Price Predictor is a machine learning-based application that estimates the price of laptops based on various hardware specifications such as brand, RAM, CPU, GPU, storage, etc. This model helps users understand potential market pricing in Indian Rupees (INR) for their desired laptop configurations.

Built using popular machine learning models, it provides an intuitive way to predict prices for new or used laptops, empowering buyers and sellers alike.

🛠️ Features

Multi-feature Input: Predicts prices based on brand, RAM, CPU, GPU, storage, and screen size.
Multiple Model Support: Leverages various machine learning models like RandomForest, GradientBoosting, XGBoost, and VotingRegressor for robust predictions.
Easy Price Display: Outputs predicted prices in Indian Rupees (INR) 💰.
Simple Interface: Easy-to-use interface where users input specifications and get a predicted price.


📊 Models Implemented
The project explores a variety of machine learning models, including:

RandomForestRegressor 🌳
GradientBoostingRegressor 📈
XGBoostRegressor 💡
ExtraTreesRegressor 🌿
VotingRegressor 🔗 (Combining multiple models for better accuracy)


🔧 Technologies & Tools

Programming Language: Python 🐍
Libraries:
pandas & numpy for data manipulation.
scikit-learn for machine learning model building.
xgboost for gradient boosting.
matplotlib & seaborn for data visualization.

Development Environment: Visual Studio Code (VS Code) 💻

🏃‍♂️ Getting Started
Prerequisites
Ensure that you have Python 3.x and the required libraries installed:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/laptop-price-predictor.git
Navigate to the project directory:

bash
Copy code
cd laptop-price-predictor
Run the predictor:

bash
Copy code
python predictor.py
🎮 How to Use
Run the program and input your laptop specifications, such as:

Brand (e.g., Dell, HP, Lenovo)
RAM (e.g., 8GB, 16GB)
CPU (e.g., Intel i5, Ryzen 5)
GPU (e.g., NVIDIA GTX 1650)
Storage (e.g., 512GB SSD, 1TB HDD)
The model will predict the price in INR based on these inputs. For example:

Input:

Brand: Dell
RAM: 16GB
CPU: Intel i7
GPU: NVIDIA GTX 1650
Storage: 512GB SSD
Predicted Price: ₹85,000

📈 Data Overview
The dataset used for this project contains laptop information with features like:

Brand: Laptop manufacturer.
RAM: Memory size in GB.
CPU: Processor type and model.
GPU: Graphics card model.
Storage: SSD or HDD capacity.
Screen Size & Resolution: Size and clarity of the display.


📅 Future Enhancements

🌐 Web App: Develop a web interface using FastAPI or Flask for broader accessibility.
📊 Comparison Feature: Add functionality to compare predicted prices with real-world prices.
📚 More Data: Extend the dataset to include more brands, models, and configurations for greater accuracy.
🏗️ Contributing


We welcome all kinds of contributions! If you’d like to improve this project, feel free to fork the repository, make changes, and submit a pull request.

📜 License
This project is licensed under the MIT License. Feel free to use and modify it as per your needs.

✉️ Contact
For any questions or suggestions, reach out to:

Md Rizwan - Data Analyst, Developer
📧 imdrizwan2019@gmail.com
🔗 LinkedIn
