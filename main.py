#import required libraries

import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pickle
from sklearn.compose import ColumnTransformer
import streamlit as st
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

#Define the main function 
#Load and Read the csv file.
#Use the Pickled file to predict
def main():
    # Streamlit app title
    st.title("Credit Risk Analyzer")

    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV file for prediction", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader("Input Data")
        st.write(df)

       # Load the model from the pickle file
        with open('model_xgb.pkl', 'rb') as file:
            data = pickle.load(file)
            preprocessor = data['preprocessor']
            model = data['model']


       # Preprocess the new data
        new_data_preprocessed = preprocessor.transform(df)

       # Predict with the loaded model
        predictions = model.predict(new_data_preprocessed)

       #Convert the predicted values to pandas dataframe
        predictions=pd.DataFrame(predictions,columns=['Predicted_Loan_Status'])

       #Concatenate with the original dataset
        final_predictions=pd.concat([df,predictions],axis=1)

       #print(final_predictions)



        #Write the Result to App
        st.write("Predicted-Status \n\n 0 - Non-Default \n\n 1 - Default") 
        st.write(final_predictions)


if __name__ == "__main__":
    #Call the main function
    main()
