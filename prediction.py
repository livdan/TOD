import argparse
import joblib
import numpy as np
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--predictor_file", type=str, required=True, default="", help="Select predictor")
    parser.add_argument("-m","--mass", type=float, required=True, help="[kg]")
    parser.add_argument("-cf","--correction_factor", type=float, required=True, choices=[0.7,0.9,1.0,1.1,1.2,1.3,1.4] , help="")
    parser.add_argument("-tr","--rectal_temperature", type=float, required=True, help="[degree C]")
    parser.add_argument("-ta","--ambient_temperature", type=float, required=True, help="[degree C]")
    args = parser.parse_args()
    
    if args.rectal_temperature < args.ambient_temperature:
        raise ValueError("Rectal temperature must be higher than the ambient temperature!")
    
    print("Predict...")
    test_data = np.zeros((1,4))
    test_data[0,0] = args.mass
    test_data[0,1] = args.correction_factor
    test_data[0,2] = args.rectal_temperature
    test_data[0,3] = args.ambient_temperature
    loaded_model = joblib.load(args.predictor_file)
    pred = loaded_model.predict(test_data)
    pred_min = pred[0]*60
    q,mod = divmod(pred[0]*60,60)
    print("Predicted TOD with" + str(loaded_model.steps[0])+": "+ format(q,".0f") + "h" +":"+ format(math.floor(mod),".0f") + "min")