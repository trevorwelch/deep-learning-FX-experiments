from sklearn.metrics import hamming_loss, classification_report, roc_auc_score, confusion_matrix, average_precision_score,log_loss,f1_score,accuracy_score,hamming_loss
import numpy as np
import pandas as pd
from keras.models import load_model



# Accepts X_train, y_train, X_test, y_test as 
def eval(model, X, y, batch_size=None, weights=None, sensitivity=0.5, temperature=1.0, binary=0, sequential=0, keras=1):


    def softmax_temp(preds, temperature=temperature):
        # helper function to sample an index from a probability array then set that index to 1 
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        final_pred_index = np.argmax(probas)
        preds.fill(0)
        preds[final_pred_index] = 1
        return preds

    if keras == 1 and weights != None:
        print("-----------------------------")
        print("-----------------------------")
        print("-----------------------------")
        model.load_weights(weights)
        print("Using:", weights)
        print("Sensitivity:", sensitivity)

    if batch_size:
        batch_size=batch_size
    elif batch_size == None:
        batch_size=1024
        

    # Preds depends on your output layer, this one is for Softmax
    y_pred_proba = model.predict(X, batch_size=batch_size)
    preds = y_pred_proba


        
    if binary == 1:
        preds = preds > sensitivity
    elif binary == 0: 
        # Take the max of a given prediction row as the true prediction
        #preds = (preds == preds.max(axis=1)[:,None]).astype(int)
        # Use the softmax_temp to decide the prediction

        # For lining up predictions and labels and other processing let's make a dataframe version as well
        df_predictions = pd.DataFrame(preds)
        df_predictions = df_predictions.iloc[:, 0:].apply(softmax_temp, axis=1)

    
    # For lining up predictions and labels and other processing let's make a dataframe version as well
    df_predictions = pd.DataFrame(preds)
    if sequential == 1:
        # Predict classes and proba ... doesn't work with functional API? 
        pred_classes = model.predict_classes(X)
        print("\n")
        y_pred_proba = model.predict_proba(X)
        print("\n")

    # summarize the fit of the model
    print("Your classification report: ")
    print(classification_report(y, df_predictions))
    print("-----------------------------")
    current_log_loss = log_loss(y, y_pred_proba)
    print("Log loss:", current_log_loss)
    f1_weighted = f1_score(y, df_predictions, average='weighted')
    print("F1 weighted:", f1_weighted)
    print("F1 macro:", f1_score(y, df_predictions, average='macro'))
    print("F1 micro:", f1_score(y, df_predictions, average='micro'))
    print("Accuracy score:", accuracy_score(y, df_predictions.astype(int)))
    hamming = hamming_loss(y, df_predictions)
    print("Hamming Loss:", hamming)

    if binary == 1:
        #print("Average precision score", average_precision_score(y, df_predictions))
        # summarize the fit of the model
        roc_auc = roc_auc_score(y, y_pred_proba)
        print("ROC AUC SCORE:", roc_auc)
        print("Your confusion matrix: ")
        print(confusion_matrix(y, df_predictions))
        print("-----------------------------")
    
    return df_predictions, y_pred_proba
    
   

# Helper function to generate predictions
def pred(model, X, batch_size=None, weights=None, sensitivity=0.5, temperature=1.0, binary=0, sequential=0, keras=1):


    def softmax_temp(preds, temperature=temperature):
        # helper function to sample an index from a probability array then set that index to 1 
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        final_pred_index = np.argmax(probas)
        preds.fill(0)
        preds[final_pred_index] = 1
        return preds

    if keras == 1 and weights != None:
        print("-----------------------------")
        print("-----------------------------")
        print("-----------------------------")
        model.load_weights(weights)
        print("Using:", weights)
        print("Sensitivity:", sensitivity)

    if batch_size:
        batch_size=batch_size
    elif batch_size == None:
        batch_size=128
        

    # Preds depends on your output layer, this one is for Softmax
    y_pred_proba = model.predict(X, batch_size=batch_size)
    preds = y_pred_proba


        
    if binary == 1:
        preds = preds > sensitivity
        # For lining up predictions and labels and other processing let's make a dataframe version as well
        df_predictions = pd.DataFrame(preds)
    elif binary == 0: 
        # Take the max of a given prediction row as the true prediction
        #preds = (preds == preds.max(axis=1)[:,None]).astype(int)

        # For lining up predictions and labels and other processing let's make a dataframe version as well
        df_predictions = pd.DataFrame(preds)
              
        # Use the softmax_temp to decide the prediction
        df_predictions = df_predictions.iloc[:, 0:].apply(softmax_temp, axis=1)

    
    
    if sequential == 1:
        # Predict classes and proba ... doesn't work with functional API? 
        pred_classes = model.predict_classes(X)
        print("\n")
        y_pred_proba = model.predict_proba(X)
        print("\n")

    return df_predictions, y_pred_proba
    
    
    
    
