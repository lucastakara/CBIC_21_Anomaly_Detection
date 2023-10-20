# Autoencoders for Anomaly Detection 

This repository explores the cutting-edge field of anomaly detection using deep learning, 
particularly through the implementation of autoencoders. Paper regarding the code 
is in the following link: https://sbic.org.br/wp-content/uploads/2021/09/pdf/CBIC_2021_paper_37.pdf
## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Python 3.10, which is required to create the virtual environment.
- You have a basic understanding of Python virtual environments.

## Setting Up and Running the Project

To set up the Python environment and run the project, follow these steps:

### 1. Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/lucastakara/CBIC_21_Anomaly_Detection.git
```

### 2. Set Up the Python Virtual Environment
Navigate to the project directory and create a virtual environment using Python 3.10. 
The following steps are for Unix-based systems like Linux and macOS. 
If you are using Windows, the commands may differ slightly.
```bash
cd /path/to/your/repository  # Navigate to the cloned repository

python3.10 -m venv venv  # Create a virtual environment named 'venv'
```
Activate the virtual environment. The command to do this varies by operating system.

* On macOS and Linux:
```bash
source venv/bin/activate
```
* On Windows (use either Command Prompt or PowerShell):
```bash
.\venv\Scripts\activate
```
### 3. Install the Dependencies
With the virtual environment activated, install the project's dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Project
Now that the environment is set up, you can run the project using:
```bash
python main.py
```

### Next implementations
[] Implement K-fold Cross-Validation
[] Perform loss reconstruction with different loss functions (MSE, MAPE, RMSE)
[] Implement Hyperparameter tuning 
