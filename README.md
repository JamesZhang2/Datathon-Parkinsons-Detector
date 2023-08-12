# Cornell Data Science Datathon Spring 2023

## Authors

- James Zhang
- Abby Kim
- Lisa Li
- Srisha Gaur

## Introduction and Background

Parkinson's Disease is a neurological disorder that affects millions of people worldwide. Typical symptoms of Parkinson’s Disease include uncontrollable tremors, impaired balance and coordination, and rigidity in the muscles. Early detection of the disease can lead to better management and treatment, which is why there is a growing interest in developing accurate and efficient methods for diagnosis.

The early diagnosis of Parkinson's Disease is a challenging task, as it involves subtle changes in the patient's motor and cognitive abilities. Currently, diagnosing Parkinson's disease relies heavily on genetic indicators, but only 15% of those with Parkinson's have a family history of the disease. Traditional diagnostic methods for Parkinson's involve invasive procedures, such as spinal taps or PET scans, making testing difficult and costly.

## What we did

This project aims to develop a Parkinson's Disease detection system using a subject's drawings. The system utilizes Computer Vision techniques to analyze and identify patterns in the drawings that are indicative of Parkinson's Disease. We used a Convolutional Neural Network (CNN) as our model, and we created a simple website in which a user downloads an image to trace and then uploads the tracing to be assessed for potential Parkinson’s symptoms. The dataset we used in our training is [HandPD](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/), which contains drawings by both healthy subjects and those with Parkinson’s. By utilizing this non-invasive and cost-effective approach, the system makes Parkinson's Disease testing more accessible to people who may not otherwise have access to expensive diagnostic tests. Ultimately, this project benefits people by providing a quicker and more accurate way to diagnose Parkinson's Disease, leading to earlier interventions and improved patient outcomes.

## Disclaimer

Please note that this model is intended for informational purposes only and may not be accurate or up-to-date. The model is not intended to replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of this medical model.

## Running our project

To run our project, please follow the instructions below:

- `git clone https://github.com/JamesZhang2/Datathon.git`
- `cd Datathon`
- Create a virtual environment: `python -m venv venv`
- Activate the virtual environment: `source venv/bin/activate` on Mac, `venv\Scripts\activate` on Windows
- Install dependencies: `pip install -r requirements.txt`
- Run frontend: `streamlit run home.py`

## Acknowledgements

Parts of the introduction and code are written with the help of ChatGPT.
