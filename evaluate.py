#######################################################################################################
# CASA--DIALOGUE-ACT-CLASSIFIER
# File contains function and main code to evaluate a saved model on the test data.
# Prior to using the evaluation function, a model must be stored.
#######################################################################################################
from typing import List, Dict

import mlflow
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report ,ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
from pandera.typing import DataFrame
import numpy as np
import matplotlib.pyplot as plt

import data.load_data as data_loader
from dataframe_schema import DataSchema
from dataset import DADataset

MLFLOW_EXPERIMENT_NAME = "CASA-dialogue-ACT-classifier"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

###############################################################################################

def evaluate_model(
        test_data:pd.DataFrame,
        label_list:List[str],
        labels:Dict[str,int],
        mlflow_model_name:str,
        mlflow_model_version:int,
        MLFlow_run_id=None) -> None:
    """_summary_

    Args:
        test_data (pd.DataFrame): _description_
        label_list (List[str]): _description_
        labels (Dict[str,int]): _description_
        mlflow_model_name (str): _description_
        mlflow_model_version (int): _description_
        MLFlow_run_id (_type_, optional): _description_. Defaults to None.
    """

    
    test_dataset = DADataset(DataFrame[DataSchema](test_data),label_dict=labels)
    test_dataloader = DataLoader(dataset=test_dataset,shuffle=False, num_workers=1,batch_size=len(test_dataset))
    
    # load the model
    model_uri = f"models:/{mlflow_model_name}/{mlflow_model_version}"
    model = mlflow.pytorch.load_model(model_uri)

    # evaluate model
    model.eval()
    with torch.no_grad():
        data_batch, intents, y_true = next(iter(test_dataloader))
        predictions = model(data_batch)
        softmax_predictions = torch.nn.functional.softmax(predictions,dim=1)        
        y_pred = torch.argmax(softmax_predictions, 1).tolist()

        # calculate scores
        accuracy = accuracy_score(y_true,y_pred,normalize=True)
        bal_accuracy_score = balanced_accuracy_score(y_true,y_pred)
        report = classification_report(y_true,y_pred,labels=range(len(label_list)),target_names=label_list,zero_division=0)
        print(f"accuracy score: {accuracy}")
        print(f"balanced_accuary_score {bal_accuracy_score}")
        print(report)

        # create confusion matrix
        plt.rcParams.update({'font.size': 6})
        fig, ax = plt.subplots(figsize=(10,10))
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                        labels=list(range(len(label_list))),
                        display_labels=label_list,
                        xticks_rotation ='vertical',
                        ax=ax,
                        #cmap='Greys',
                        colorbar=False)
       
        plt.subplots_adjust(left=0.3, right=0.99, bottom=0.3, top=0.99)
        plt.show()


        # plot distribution
        cm = disp.confusion_matrix # type: ignore
        TP = cm.diagonal()
        total = cm.sum(axis=0)
        total[total == 0] = 1 # prevent division by zero
        FP = total - TP
        percent = np.nan_to_num(TP/total * 100)
        fig, ax = plt.subplots(figsize=(20,10))

        rects1 = ax.bar(label_list, total, 0.5, label='total occurence in test set',color="lightgrey")
        rects2 = ax.bar(label_list, TP, 0.5, label='True Positives',color="royalblue")

        for i, v in enumerate(percent):
            ax.text(i,TP[i] + 10, str(round(v,1)) + "%", color='blue',horizontalalignment="center")

        for label in ax.get_xticklabels():
                label.set_rotation(90) 
        ax.set_ylabel('count')
        ax.set_xlabel('intent class')
        plt.legend()
        plt.title("class distribution and true positive predictions")
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.3, top=0.99)
        plt.show()
        

        # save scores to experiment if experiment run MLFlow_run_id is provided
        if MLFlow_run_id:
            with mlflow.start_run(run_id=MLFlow_run_id):
                report = classification_report(y_true,y_pred,labels=label_list,zero_division=0,output_dict=True)
                
                # get last step index and stor values
                last_epoch = len(mlflow.MlflowClient().get_metric_history(MLFlow_run_id,"validation loss"))-1 
                mlflow.log_figure(fig, f'TP_distribution_{last_epoch}.png')
                mlflow.log_metric("accuracy", float(accuracy),step=last_epoch)
                mlflow.log_metric("balanced accuracy", bal_accuracy_score ,step=last_epoch)
                mlflow.log_dict(report,f"classification_report_epoch_{last_epoch}.json")
                mlflow.log_figure(disp.figure_, f'confusion_matrix_epoch_{last_epoch}.png')

        

###############################################################################################            

if __name__=="__main__":
    
    DATA_FOLDER = "data"
    # model name and version as stored in MLFLow
    MODEL_NAME = "CASA-dialogue-ACT-classifier-V1.0"
    MODEL_VERSION = 6
    
    # add run id to register evaluation results to the experiment
    #RUN_ID = "382131c84d8a445cbe511620d3b7a27f" #None
    RUN_ID = None

    # load the data
    test_data = data_loader.get_test_data(DATA_FOLDER)
    test_data['utterance'] = test_data['utterance'].fillna(value="")
    label_list = data_loader.get_label_list(DATA_FOLDER)
    labels = data_loader.get_labels(DATA_FOLDER)
    
    # evaluate the model
    evaluate_model(test_data,
                    label_list=label_list,
                    labels = labels,
                    mlflow_model_name=MODEL_NAME,
                    mlflow_model_version = MODEL_VERSION,
                    MLFlow_run_id= RUN_ID
                   )