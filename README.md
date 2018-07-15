# student_poisson_mixture
----------------------------------

The code is released for the paper [Understanding Student Procrastination vis Mixture Models](), EDM 2018

Jihyun Park (`jihyunp@ics.uci.edu`)<br>
July 2018

## Required Packages
Written in `Python2.7`.<br>
Python packages `numpy`, `scipy`, `random`, and `matplotlib` are needed to run the code.


## Data
- `test_data.csv`: 
Sample data (simulated data) to fit the Poisson mixture model.
Each row in the file is considered as a daily activity count vector for a student.
400 rows exist in this sample data.


## Demo iPython Notebook
- `demo.ipynb`: 
A quick tutorial of using the code.


## Code
- `pmm.py`:
   Code for fitting Poisson mixture model given a count matrix.
   The file has two classes--`PoissonMixture` for the model and `PoisMixResult` for 
   storing and plotting the result.  
- `utils.py`:
   Has helper functions for calculating log probabilities.