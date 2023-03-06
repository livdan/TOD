import run.dataset as ds
import numpy as np
import csv
import os
import shutil

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from time import time
from sklearn.pipeline import make_pipeline
from joblib import dump

class TOD:

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
  
    def CreatedDatasetSVR(self,dir,svmfilename,generated_data):
        dataset = ds.Dataset()
        svmpath = dir+'/'+svmfilename
        dataset.CreateDataset(dir,svmpath,generated_data)
        return dataset.LoadDataset(svmpath)

    def CreateTrainTestData(self,dataLines):
        X=[]
        y=[]

        for dl in dataLines:
            dl = eval(dl)
            y.append(float(dl['delta_time']))
            X.append([float(dl['weight']), float(dl['corrfactor']), float(dl['measured_temp']), float(dl['ambient_temp'])])

        return X,y

class Predict:

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def FitWithGridSearchCV(self,name,maindata_dir,X_train, y_train, X_test, y_test,pipe,param_grid,train_count,verbose,cv) -> object:           
        
        grid_search = GridSearchCV(pipe, param_grid=param_grid,verbose=verbose,cv=cv,n_jobs=-1)
    
        start = time()
        
        grid_result = grid_search.fit(X_train, y_train)
    
        train_time = time() - start
        start = time()
            
        r_predict= np.zeros((len(y_test)))
        err= np.zeros((len(y_test)))
        y_test_temp = np.zeros((len(y_test)))
          
        for ri in range(len(y_test)):
            tmp_m = X_test[ri].copy()      
            r_predict[ri] = grid_search.predict([tmp_m])            
            err[ri] = r_predict[ri]-float(y_test[ri])
            y_test_temp[ri] = y_test[ri]
         
        predict_time = time()-start    

        print("\tTraining time: %0.3fs" % train_time)
        print("\tPrediction time: %0.3fs" % predict_time)
        print("\tMean absolute error:", mean_absolute_error(y_test, r_predict))
        print("\tMean squared error:", mean_squared_error(y_test, r_predict))
        print("\tR2 score:", r2_score(y_test, r_predict))
        print("\tThe best parameters are:", str(grid_result.best_params_))
        print("\twith a score of:", str(grid_result.best_score_))
        print()
            
        dir = str(maindata_dir)+'/result_'+str(name)
     
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        #save model:
        estimator = grid_result.best_estimator_                
        dump(estimator, dir+'/'+str(name)+'.joblib')      
               
        f = open(dir+'/'+str(name)+'.csv', 'w', newline='')
        f.writelines([str(name),"\nTraining time [s]: " + format(train_time,'0.3f'),
                          "\nPrediction time [s]: " + format(predict_time,'0.3f'),
                          "\nMean absolute error: " + str(mean_absolute_error(y_test, r_predict)),
                          "\nMean squared error: " + str(mean_squared_error(y_test, r_predict)),
                          "\nR2 score: " + str(r2_score(y_test, r_predict)),
                          "\nThe best parameters are:" + str(grid_result.best_params_),
                          "\nwith a score of:" + str(grid_result.best_score_)])
        f.close()
           
        fieldnames = ['train_count','name','trainingtime','predictiontime','mean_absolute_error','mean_squared_error','R2']
        
        resultfile = str(maindata_dir)+'/result.csv'     
        if not os.path.exists(resultfile):    
            with open(resultfile, 'w', newline='') as csvfile:   
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
       
        with open(resultfile, 'a', newline='') as csvfile:    
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)                                           
            writer.writerow({'train_count': train_count, 'name': str(name),'trainingtime' : format(train_time,'0.3f'),'predictiontime' : format(predict_time,'0.3f'),'mean_absolute_error' : str(mean_absolute_error(y_test, r_predict)),'mean_squared_error': str(mean_squared_error(y_test, r_predict)),'R2' : str(r2_score(y_test, r_predict))})
            
        with open(dir+'/'+str(name)+'_result.csv', 'w', newline='') as csvfile:
            fieldnames = ['m','corrfactor','measured_temp', 'Ta','expected_t','predicted_t', 'error']
                
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
                        
            for i in range(len(X_test)):            
                writer.writerow({'m': format(X_test[i][0], '.1f'),'corrfactor': X_test[i][1],'measured_temp': format(X_test[i][2],'.1f'),'Ta': format(X_test[i][3],'.1f'),'expected_t': format(y_test_temp[i],'.1f'), 'predicted_t':  r_predict[i], 'error': err[i]})

        return estimator
    
    
    def PredictTOD(self,maindata_dir,dataLines,calculate_decisiontrees,calculate_svm,verbose,cv):          
        X,y = TOD().CreateTrainTestData(dataLines)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        valid_train_count = len(X)

        if calculate_decisiontrees.lower() == "true":
            
            print("Var 1.: DecisionTreeRegressor:")       
            decisiontreeregressor_pipe = make_pipeline(DecisionTreeRegressor())        
            decisiontreeregressor_param_grid = {
                "decisiontreeregressor__splitter": ["best", "random"],
                "decisiontreeregressor__max_features" : [None, "sqrt", "log2"],
                "decisiontreeregressor__criterion" :["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "decisiontreeregressor__random_state" : [None,10,25,50,100,150]
                }
            
            self.FitWithGridSearchCV(name="decisiontreeregressor",
                                    maindata_dir=maindata_dir,
                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,                                 
                                    pipe=decisiontreeregressor_pipe,param_grid=decisiontreeregressor_param_grid,
                                    train_count = valid_train_count,
                                    verbose = verbose,
                                    cv = cv)
            
#----------------------------------------------------            
            
            print("Var. 2: RandomForestRegressor")       
            randomforestregressor_pipe = make_pipeline(RandomForestRegressor())        
            randomforestregressor_param_grid = {
                "randomforestregressor__n_estimators": [20,50,100,125,150,200],
                "randomforestregressor__max_features" : [None, "sqrt", "log2"],
                "randomforestregressor__criterion" :["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "randomforestregressor__random_state" : [None,10,25,50,100]
                }
            
            self.FitWithGridSearchCV(name="randomforestregressor",
                                    maindata_dir=maindata_dir,
                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,                                 
                                    pipe=randomforestregressor_pipe,param_grid=randomforestregressor_param_grid,
                                    train_count = valid_train_count,
                                    verbose = verbose,
                                    cv = cv)
            
#----------------------------------------------------   
            
            print("Var. 3: ExtraTreesRegressor")       
            extratreesregressor_pipe = make_pipeline(ExtraTreesRegressor())        
            extratreesregressor_param_grid = {
                "extratreesregressor__n_estimators": [20,50,100,125,150,200],
                "extratreesregressor__max_features" : [None, "sqrt", "log2"],
                "extratreesregressor__criterion" :["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "extratreesregressor__bootstrap" : [True,False]
                }
            
            self.FitWithGridSearchCV(name="extratreesregressor",
                                    maindata_dir=maindata_dir,
                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,                                 
                                    pipe=extratreesregressor_pipe,param_grid=extratreesregressor_param_grid,
                                    train_count = valid_train_count,
                                    verbose = verbose,
                                    cv = cv)

#----------------------------------------------------   
            
            print("Var. 4: BaggingRegressor")       
            baggingregressor_pipe = make_pipeline(BaggingRegressor())        
            baggingregressor_param_grid = {
                "baggingregressor__n_estimators": [10,20,50,100,125,150,200],
                "baggingregressor__bootstrap" : [True,False]              
                }
            
            self.FitWithGridSearchCV(name="baggingregressor",
                                    maindata_dir=maindata_dir,
                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,                                 
                                    pipe=baggingregressor_pipe,param_grid=baggingregressor_param_grid,
                                    train_count = valid_train_count,
                                    verbose = verbose,
                                    cv = cv)                
                
#---------------------------------------------------- SVM ----------------------------------------------------
        if calculate_svm.lower() == "true":
            print("Var. 8: SVR")       
            svr_pipe = make_pipeline(StandardScaler(),SVR(kernel="rbf"))        
            svr_param_grid = {
                "svr__epsilon": [0.005, 0.01, 0.05, 0.1],
                "svr__C" : [0.1,1,2,5,10,20,50,100],
                "svr__gamma": [1,2,3,4,5]
                }
            
            estimator_svr = self.FitWithGridSearchCV(name="svr",
                                    maindata_dir=maindata_dir,
                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,                                 
                                    pipe=svr_pipe,param_grid=svr_param_grid,
                                    train_count = valid_train_count,
                                    verbose = verbose,
                                    cv = cv)
            
            print("Var. 9: AdaBoostRegressor with SVR")       
            adaboostregressor_pipe = make_pipeline(AdaBoostRegressor())        
            adaboostregressor_param_grid = {
                "adaboostregressor__estimator" : [estimator_svr],
                "adaboostregressor__loss": ["linear", "square", "exponential"],
                "adaboostregressor__n_estimators" : [20,50,75],
                "adaboostregressor__random_state" : [None,2,5]
                }
            
            self.FitWithGridSearchCV(name="adaboostregressor_svr",
                                    maindata_dir=maindata_dir,
                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,                                 
                                    pipe=adaboostregressor_pipe,param_grid=adaboostregressor_param_grid,
                                    train_count = valid_train_count,
                                    verbose = verbose,
                                    cv = cv)
