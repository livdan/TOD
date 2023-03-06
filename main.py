import argparse
import run.todmain as tod
import myutils.generator as dg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt","--decisiontrees", type=str, required=False, default="True", help="Calculate with decision trees")
    parser.add_argument("-svm","--svm", type=str, required=False, default="True", help="Calculate with SVM")
    parser.add_argument("-dir","--maindata_dir", type=str, required=False, default="maindata", help="Path to data dir (without /)")
    parser.add_argument("-c","--count", type=int, required=False, default=3000, help="Count of main generated data")
    parser.add_argument("-s","--sigma", type=int, required=False, default=10, help="Value of sigma")
    parser.add_argument("-v","--verbose", type=int, required=False, default=2, help="Controls the level of logging output during model training")
    parser.add_argument("-cv","--cross_validation", type=int, required=False, default=2, help="Split the dataset into multiple 'folds'")
    args = parser.parse_args()
        
    print("Create dataset...")
    generated_data= dg.Datagenerator().GenerateData(args.count,args.maindata_dir,args.sigma) 
    svm_datalines = tod.TOD().CreatedDatasetSVR(args.maindata_dir+'/data','labeled_dataset_.csv',generated_data)
    
    print("Create dataset, ready. Begin predict...")
    tod.Predict().PredictTOD(args.maindata_dir,svm_datalines,calculate_decisiontrees = args.decisiontrees,calculate_svm = args.svm,verbose=args.verbose,cv=args.cross_validation)
    print("Predict, ready.")   
