# Human-Behavior-Digital-Wellbeing-Dataset-Generator

A Python-based toolkit designed to generate, validate, and analyze synthetic datasets focused on digital device usage, user behavioral patterns, and mental/physical well-being metrics.

---

## Features

- **Synthetic Data Generation:** Produce realistic synthetic data modeling screen time, notification frequency, app usage, stress scores, sleep quality, and other digital-wellbeing indicators using `generate.py`.

- **Data Validation:** Ensure datasets adhere to expected structural schemas, non-negative bounds, data types, and logical constraints using `validate.py`.

- **Exploratory Analytics:** Includes a comprehensive Jupyter Notebook (`human-digital-behavior-wellbeing-analytics.ipynb`) for statistical analysis, trend visualization, behavioral analysis, and correlation studies.

- **Reproducible Dataset Generation:** Generate new synthetic datasets programmatically using a configurable Python-based pipeline.

- **Data Quality Checks:** Validate generated datasets before using them for machine learning, analytics, or research purposes.

---

## Repository Structure

```text
Human-Behavior-Digital-Wellbeing-Dataset-Generator/
│
├── Notebooks/
│   └── human-digital-behavior-wellbeing-analytics.ipynb
│
├── generate.py
├── validate.py
├── requirements.txt
├── LICENSE
└── README.md
```

### File Description

| File / Directory | Description |
|---|---|
| `Notebooks/` | Contains the exploratory data analysis notebook |
| `generate.py` | Generates the synthetic dataset |
| `validate.py` | Validates dataset structure and data quality |
| `requirements.txt` | Lists required Python dependencies |
| `LICENSE` | Project license |
| `README.md` | Project documentation |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MobeenFatimaa/Human-Behavior-Digital-Wellbeing-Dataset-Generator.git
```

### 2. Navigate to the Project Directory

```bash
cd Human-Behavior-Digital-Wellbeing-Dataset-Generator
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Generate Data

Run the generator script to create a fresh synthetic dataset:

```bash
python generate.py
```

The generated dataset can then be used for exploratory analysis, visualization, statistical modeling, and machine learning experiments.

---

### Validate Data

Run the validation script to check the generated dataset for structural and logical integrity:

```bash
python validate.py
```

The validation pipeline checks whether the dataset follows the expected schema and predefined constraints.

---

### Analyze the Dataset

Launch the Jupyter Notebook to explore the generated data:

```bash
jupyter notebook Notebooks/human-digital-behavior-wellbeing-analytics.ipynb
```

The notebook provides statistical summaries, visualizations, behavioral trends, and correlation analysis.

---

## Dataset Applications

This synthetic dataset can be used for:

- Exploratory Data Analysis (EDA)
- Machine Learning experiments
- Behavioral pattern analysis
- Digital wellbeing research
- Data visualization projects
- Statistical modeling
- Feature engineering
- Predictive modeling
- Educational data science projects
- Synthetic data generation research

---

## Key Research Areas

The dataset is designed around relationships between:

- Digital device usage
- Screen time
- Notification frequency
- Application usage
- Sleep quality
- Stress levels
- Physical activity
- Behavioral patterns
- Mental and physical wellbeing indicators

These relationships can be explored to identify trends, correlations, behavioral patterns, and potential predictive features.

---

## Data Validation

The `validate.py` script provides an additional quality-control layer before the dataset is used for analysis or modeling.

Validation can include checks for:

- Required columns
- Correct data types
- Missing values
- Non-negative values
- Valid numerical ranges
- Logical relationships between variables
- Dataset structure and consistency

---

## Exploratory Data Analysis

The included notebook provides an interactive environment for investigating the generated dataset.

Typical analysis includes:

- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Behavioral trends
- Wellbeing comparisons
- Feature relationships
- Data visualization

---

## Requirements

The project uses Python and common data-science libraries.

Install all dependencies using:

```bash
pip install -r requirements.txt
```

A virtual environment is recommended:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Synthetic Data Disclaimer

This project generates **synthetic data** for research, experimentation, education, and machine learning purposes.

The generated records do not represent real individuals and should not be interpreted as medical, psychological, or clinical evidence.

The dataset should not be used to make real-world medical or mental-health decisions.

---

## Project Goals

The primary goals of this project are to:

1. Generate realistic synthetic digital-wellbeing data.
2. Provide reliable validation mechanisms.
3. Support reproducible data-science experiments.
4. Enable exploratory behavioral analysis.
5. Provide a foundation for machine-learning projects.
6. Demonstrate practical synthetic-data generation techniques.

---

##  Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Jupyter Notebook**
- **Matplotlib**
- **Seaborn**
- **Synthetic Data Generation**
- **Data Validation**
- **Exploratory Data Analysis**

---

##  Example Workflow

```text
Generate Synthetic Data
        │
        ▼
    Validate Data
        │
        ▼
 Exploratory Analysis
        │
        ▼
 Feature Engineering
        │
        ▼
 Machine Learning
        │
        ▼
 Insights & Visualization
```

---

##  Contributing

Contributions, suggestions, and improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Push the branch.
6. Open a Pull Request.

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

## Author

**Mobeen Fatima**

Computer Science (Specialized AI) Student | AI & Data Science Enthusiast

GitHub: [MobeenFatimaa](https://github.com/MobeenFatimaa)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
