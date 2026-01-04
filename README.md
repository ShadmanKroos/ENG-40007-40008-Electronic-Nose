# ENG-40007-40008-Electronic-Nose

![IMG_7811](https://github.com/user-attachments/assets/b641eb05-3f2e-44a1-8608-d9ec346c2286)


Project Overview:

This repository contains the source code, datasets, and documentation for a Final Year Capstone Project focused on developing a low-cost Electronic Nose (E-Nose) system. The project aims to mimic biological olfactory functions using artificial sensor arrays and machine learning (ML) algorithms to effectively classify distinct spice odors.
Utilising the Bosch BME688 metal oxide (MOX) gas sensor platform, this system addresses key challenges in current E-Nose technology, such as sensor drift, environmental sensitivity, and the lack of standardised datasets. The proof-of-concept prototype successfully discriminates between complex olfactory patterns of four spice specimens: Anise, Chilli, Cinnamon, and Nutmeg.

Key Objectives:

• Hardware Integration: To design a portable data acquisition setup using the BME688 Development Kit and ESP32 microcontroller to capture Volatile Organic Compound (VOC) signatures.

• Standardised Methodology: To establish a reliable protocol for prolonged multi-sensor odor data acquisition (6-hour sessions) under varying environmental conditions.

• Machine Learning Analysis: To develop and evaluate supervised learning models (including Random Forest, XGBoost, and Gradient Boosting) to classify odor profiles with high accuracy.

• Performance Evaluation: To assess the impact of raw versus feature-engineered data on classification robustness.
Hardware Specifications

The system is built upon the Bosch BME688 Development Kit, which integrates eight metal-oxide (MOX) gas sensors capable of detecting VOCs, VSCs, and other gases in the parts per billion range.

• Sensor Module: BME688 DevKit (8 x MOX sensors).

• Microcontroller: Adafruit HUZZAH32 (ESP32).

• Key Features: Integrated temperature, pressure, and humidity sensors for environmental compensation, along with VOC specific resistance for each specimen.

• Configuration: Uses the default Heater Profile (HP-354) and Duty Cycle (RDC-5-10) to modulate sensor temperature and resistance profiling.

Methodology and Data Acquisition:

To ensure dataset robustness and address the "sensor drift" gap identified in literature, data collection was conducted as follows:

• Specimens: Four distinct spices (Anise, Chilli, Cinnamon, Nutmeg) placed in airtight glass jars.

• Sampling Duration: Continuous 6-hour acquisition sessions across four different days to capture temporal and environmental variability.

• Data Volume: The final trimmed dataset consists of approximately 54,400 rows of data, ensuring a standardised input for ML training.


Data Shuffling and Structuring:

To address irregularities where the raw BME688 sensor output did not strictly maintain the expected "Sensor Index" hierarchy, a specific data shuffling protocol was implemented using Python. This process reorganized the converted `.csv` data to enforce a consistent nested looping pattern based on three hierarchical indices:

1.  **Sensor Index (Inner Loop):** Iterates 0 to 7.
2.  **Heater Profile Step Index (Middle Loop):** Iterates 0 to 9, incrementing only after all eight sensors are measured.
3.  **Scanning Cycle Index (Outer Loop):** Iterates 1 to 5, incrementing after all heater steps are completed.

This shuffling ensured that every block of 400 rows represented exactly one complete execution of the scanning cycle (5 cycles × 10 heater steps × 8 sensors), allowing for the precise trimming of out-of-pattern data caused by power-off events.

Machine Learning Models and Results:

Ten different machine learning algorithms were developed and compared using Google Colab which are Random Forest, Logistic Regression, SVM, SGD, MLP, CNN, KNN, Adaboost, GBDT, and XGBoost.
Key Findings:

• Best Performance: Ensemble models trained on raw sensor data (all four features: Resistance, Pressure, Humidity, Temperature) achieved the highest accuracy.

    - Gradient Boosting: 81.09% Accuracy.
    - XGBoost: 81.06% Accuracy.
    -  Random Forest: 80.66% Accuracy.
    
• Class-Specific Accuracy: The models achieved near 100% recall for Chilli and Nutmeg, though Cinnamon proved more challenging to classify due to misclassification as Anise.

• Feature Importance: Surprisingly, models using Pressure data only performed remarkably well, with Logistic Regression achieving 79.39% accuracy, outperforming temperature or resistance-only models in isolation

Challenges & Technical Insights:

A critical outcome of this research was the evaluation of feature engineering versus raw data utilization.

• Impact of Pre-Processing: The project developed three pre-processed datasets ("Reduced", "Reduced Plus", and "Reduced Plus 2") incorporating baseline normalisation and statistical feature extraction,. However, experimental results demonstrated that these techniques resulted in the loss of essential odor-specific patterns, leading to lower classification accuracies (max 70%) compared to raw data models (max 81%).

The raw BME688 sensor outputs-specifically the combination of resistance, pressure, temperature, and humidity-contain complex, non-linear dependencies that ensemble algorithms (like Gradient Boosting and XGBoost) interpret more effectively than human-engineered statistical features.
