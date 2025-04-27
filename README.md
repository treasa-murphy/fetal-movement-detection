## **Saving Babies With Machine Learning and Time Series Analysis**

![Fetal Movement Monitoring](images/fyp-fmm-pregnancy.png)

### **Overview**

This project investigates the application of machine learning and time series classification techniques for fetal movement detection using wearable sensor data. The dataset, collected by clinicians at a maternity hospital in Ireland, was recorded using the FeMo (Fetal Movement Monitoring) belt, a non-invasive wearable system equipped with piezoelectric sensors. It includes sensor signals alongside maternal button-press annotations marking perceived fetal movements. Accurate detection of fetal movements is critical for prenatal care, providing early indicators of fetal health and wellbeing. The goal of this work is to contribute towards the development of lightweight, scalable, and non-invasive prenatal monitoring systems, enabling more accessible and proactive maternal and fetal healthcare.


![Fetal Movement Monitoring](images/femo-belt.jpeg)

**Figure 1.** Hardware system for the wearable fetal movement monitor. (a) Sensors embedded in an elastic belt, (b) belt worn by a pregnant participant, (c) CAD design of the custom-made acoustic sensor, and (d) miniaturised (62 mm × 31 mm) DAQ system designed for the FM monitor.  

*Image adapted from Ghosh et al., 2024. [Source](https://www.sciencedirect.com/science/article/pii/S1566253523004402).*

---

### **Dataset**

The dataset was collected by clinicians at a maternity hospital in Ireland and involved 40 participants at 36 to 40 weeks of gestation. Data was recorded using the FeMo wearable monitoring system, with particular focus on belts A and C, which incorporated large piezoelectric sensors positioned at p1 and p4. For this study, only the supervised hospital sessions were used, where participants manually annotated perceived fetal movements by pressing a button. These annotations served as the ground truth for model training and evaluation.

---

### **Project Objectives**

- Explore the use of state-of-the-art time series classification techniques (Quant, Rocket, Hydra) for fetal movement detection.
- Develop an efficient, lightweight pipeline optimised for minimalistic wearable sensor data.
- Investigate the impact of different sample generation strategies and class balancing techniques on classification performance.
- Evaluate model performance using participant-independent splits to ensure generalisability.
- Assess whether accurate movement detection is achievable using piezoelectric sensor data (p1 and p4), without the need for complex sensor fusion.
- Identify challenges and opportunities for future research towards scalable, non-invasive prenatal monitoring systems.

---

### **Methodology**

- **Data Preprocessing:**  
  - Cleaned and validated raw sensor data.
  - Focused on signals from piezoelectric sensors p1 and p4.
  - Excluded noisy segments (e.g., belt adjustment periods).

- **Sample Generation Strategies:**  
  - **Strategy 1:** Non-overlapping 5-second windows centred around maternal button-clicks.
  - **Strategy 2:** Non-overlapping 5-second windows across entire sessions, labelled based on presence of a button-click.
  - **Strategy 3:** Introduced overlapping positive windows to augment movement samples and boost recall.

- **Model Training and Evaluation:**  
  - Compared state-of-the-art time series classifiers: Quant, Rocket, Hydra.
  - Selected QUANT + Scaling + LDA pipeline for focused experimentation based on initial results.

  ![Fetal Movement Monitoring](images/fmm-pipeline.png)

  - Applied class balancing techniques to address dataset imbalance (2:1 negative:positive and positive:negative setups).

- **Performance Metrics:**  
  - Evaluated models using F1-Score, average accuracy, precision, and recall.
  - Used participant-independent splits to measure generalisation performance.

---

### **Key Results**

- Accurate fetal movement classification is feasible using a minimalistic sensor configuration.
- Best-performing pipeline (QUANT + Scaling + LDA) achieved:  
  - **F1-Score:** 0.52  
  - **Average Accuracy:** 0.65  
  (using p1 sensor data with balanced training and testing)
- Incorporating p4 alongside p1 did not significantly improve generalisation compared to using p1 alone.
- Targeted sampling strategies (Strategy 1) and rigorous class balancing were critical to improving model performance.
- Overlapping positive samples (Strategy 3) boosted recall but introduced more false positives, highlighting a trade-off between sensitivity and specificity.

---

### **Project Structure**

- `/data/` — Contains raw and preprocessed datasets (not included in public repository).
- `/notebooks/` — Jupyter notebooks for exploratory analysis, sample generation, and model training.
- `/models/` — Trained model files and evaluation results.
- `/src/` — Source code including sample generation scripts, feature extraction, and classification pipelines.
- `README.md` — Project overview and documentation.
- `requirements.txt` — List of Python dependencies.

---

### **Future Work**

- Incorporate multimodal sensor fusion by combining piezoelectric and IMU sensor data.
- Extend validation to include unsupervised home recording sessions.
- Explore deep learning architectures such as CNNs and hybrid CNN-LSTM models.
- Integrate real-time movement detection capabilities.
- Validate findings using ultrasound-confirmed movement annotations.

---

### **Acknowledgements**

I would like to thank Assoc. Prof. Georgiana Ifrim for her exceptional supervision and guidance throughut this project. I am also grateful to Dr. Colin Boyle and Prof. Niamh Nowlan from the FeMo team for their biomedical engineering insights. Special thanks to the mothers who participated in the FeMo study for making this research possible.

---

### **Disclaimer**

This project was conducted solely for academic research purposes.  
The models and findings presented here are not intended for clinical use without further validation and regulatory approval.














