# Speech Emotion Recognition

## Project Overview
This project aims to revolutionize human-computer interaction by accurately identifying and interpreting emotions from speech signals. By leveraging advanced machine learning techniques and feature extraction methods, the Speech Emotion Recognition (SER) system can classify emotions such as happiness, anger, sadness, and more with high accuracy.

## Objectives
- Build a robust SER system using deep learning.
- Extract meaningful features from speech audio to understand emotional cues.
- Achieve high accuracy in emotion detection using diverse datasets.
- Explore potential real-world applications, such as customer service and mental health support.

## Key Features
- **Datasets:** Utilized RAVDESS, TESS, and CREMA-D datasets to capture a broad spectrum of emotional expressions.
- **Feature Extraction:** Implemented advanced techniques like Mel-frequency Cepstral Coefficients (MFCC), Zero-Crossing Rate (ZCR), and Root Mean Square Error (RMSE).
- **Deep Learning Model:** Designed and trained a Convolutional Neural Network (CNN) to classify emotions.
- **Data Augmentation:** Enhanced model generalization through techniques like noise addition, pitch alteration, and time-stretching.
- **Accuracy:** Achieved a testing accuracy of 92%.

## Dataset
### Sources:
1. **RAVDESS:** Ryerson Audio-Visual Database of Emotional Speech and Song.
2. **TESS:** Toronto Emotional Speech Set.
3. **CREMA-D:** Crowdsourcing Emotional Database for Speech and Song.

### Preprocessing Steps:
- Cleaned and standardized audio data.
- Extracted key features (MFCC, ZCR, RMSE) for model training.
- Applied data augmentation to increase dataset variability.

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:** TensorFlow, Keras, Librosa, Pandas, NumPy, Matplotlib, Seaborn
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Environment:** Jupyter Notebook

## Methodology
1. **Data Preprocessing:**
   - Extracted relevant audio features (MFCC, ZCR, RMSE).
   - Normalized and prepared datasets for training.

2. **Exploratory Data Analysis (EDA):**
   - Analyzed the distribution of emotions across datasets.
   - Visualized audio spectrograms and feature trends.

3. **Model Training:**
   - Designed a CNN model for emotion classification.
   - Trained the model using augmented and raw data.

4. **Evaluation:**
   - Assessed model performance using metrics like accuracy and loss.
   - Visualized learning curves over epochs.

## Results
- Achieved 92% accuracy in emotion classification.
- Demonstrated model robustness with diverse datasets and augmented data.
- Successfully categorized emotions such as happiness, anger, sadness, and disgust.

## Potential Applications
- **Customer Service:** Enhance interactions by detecting customer emotions.
- **Mental Health Monitoring:** Identify stress or anxiety patterns from speech.
- **Educational Tools:** Tailor learning experiences based on student emotions.
- **Virtual Assistants:** Adapt responses based on user emotions.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/AmishaMe24/Speech-Emotion-Recognition-System.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage
1. Load the datasets as described in the notebook.
2. Execute preprocessing steps to extract features and normalize data.
3. Train the CNN model and evaluate its performance.
4. Test the model with custom audio samples.

## Future Improvements
- Integrate additional datasets for better generalization.
- Explore transfer learning for improved performance.
- Implement real-time emotion detection capabilities.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with enhancements or bug fixes.

## Contact
For any queries, feel free to reach out:
- **Author:** Amisha Mehta
