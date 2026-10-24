# Track 2: AI Data Center Power–Computing Co-Scheduling

## Overview
The co-scheduling problem is formulated as a mixed-integer linear program  that decides, for every hour of the day, how much of each task to execute and how to power it from various sources of energy. The dashboard allows for user inputs and displays the optimization results.

## Prerequisites
* Python 3.8 or higher
* pip

## Dependencies
All required libraries (Streamlit, PuLP, Pandas, NumPy, Matplotlib, Highspy) are listed in 'requirements.txt'.

## Installation
Open a terminal and run:

### 1. Clone the repository
```bash
git clone https://github.com/caitlinzzzhou/caitlin_amy_sunny_hackshsidT2.git
```

### 2. Move into the project directory
```bash
cd caitlin_amy_sunny_hackshsidT2.git
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to run the project
Open a terminal and run:

### Running the dashboard
```bash
streamlit run datacenter_gui.py
```

### Running MILP + full report
Note: report saved in the outputs/ folder in the current directory
```bash
python main.py
```

### Running MILP vs. Greedy comparison only
```bash
python main.py --base
```
