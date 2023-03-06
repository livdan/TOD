# TOD - Determination of Time Interval of Death

## Description
This Python script is used for generating and predicting Time-of-Death (TOD) information using Support Vector Machine (SVM) and decision trees. It uses command-line arguments to customize the process.

The question of determining the time interval of death is one of the oldest and most challenging questions in forensic medicine. The process of changing the temperature of a dead body involves a complex interplay of several biological factors and is described by physical laws. The cooling curve is sigmoidal, and the traditional Newtonian cooling law is not suitable for describing the process mathematically due to a plateau phase. Empirical solutions, such as the Marshall-Hoare formula and Henssge nomogram, have been developed to address the problem.

Our data-driven model is based on generated data and provides a new perspective on the traditional methods used to determine the time interval of death. By using different regression methods, such as decision trees and SVM, we were able to estimate the time of death with high accuracy.

## Prerequisites
To use this script, you need to have Python 3.x or later installed.
This project relies on the following Python packages and their specified versions:

```
numpy>=1.24.1
scipy>=1.9.3
scikit-learn==1.2.1
joblib>=1.2.0
matplotlib>=3.6.2
```
Install the required packages by running:
```
pip install -r requirements.txt
```
## Usage - training
To use the script, run it with the following command:
```
python main.py [OPTIONS]
```
The available options are:
```
-dt / --decisiontrees: A boolean option indicating whether to calculate with decision trees or not. Default is True.
-svm / --svm: A boolean option indicating whether to calculate with SVM or not. Default is True.
-dir / --maindata_dir: The path to the data directory, without the trailing slash. Default is maindata.
-c / --count: The count of the main generated data. Default is 3000.
-s / --sigma: The value of sigma. Default is 10.
-v / --verbose: Controls the level of logging output during model training. Default is 2.
-cv / --cross_validation: Split the dataset into multiple 'folds'. Default is 2.
```
Example:
```
python main.py -dt True -svm True -dir maindata -c 5000 -s 20 -v 1 -cv 5
```
## Usage - predict
To predict the time of death run:
```
python prediction.py [OPTIONS]
```
Command-line Arguments
```
-p, --predictor_file (required): Path to the predictor file. Select the required .joblib file from maindata_dir/name_of_the_method.
-m, --mass (required): Mass of the animal in kilograms.
-cf, --correction_factor (required): Correction factor (must be one of the choices: [0.7,0.9,1.0,1.1,1.2,1.3,1.4]).
-tr, --rectal_temperature (required): Rectal temperature of the animal in degrees Celsius.
-ta, --ambient_temperature (required): Ambient temperature in degrees Celsius.
```
Example:
```
python main.py -p predictor.joblib -m 50 -cf 1.1 -tr 39 -ta 25
```
## License
MIT (see LICENSE file for details)

