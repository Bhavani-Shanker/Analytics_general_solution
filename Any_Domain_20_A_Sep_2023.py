

#######################################################
import streamlit as st
import pandas as pd
from evalml import AutoMLSearch


import evalml

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from googletrans import Translator as google_translator
import joblib
from gtts import gTTS  # pip install gtts
import os
from summa import summarizer
import glob




# Function to load data from CSV file
def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    # uploaded_file =  "BostonHousing.csv"
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
    return df,uploaded_file # Return both the dataframe
 
def convert_to_csv(dataframe):
    csv_data = dataframe.to_csv(index=False)
    return csv_data
# Create a button that, when clicked, initiates the download of a file
def download_button(label, file_content, file_name):
    download_button_str = f'<a href="data:file/txt;base64,{file_content}" download="{file_name}">{label}</a>'
    st.markdown(download_button_str, unsafe_allow_html=True)
def calculate_scores(true, pred):
   
    scores = {}
    scores = pd.DataFrame(
        {
            "R2": r2_score(true, pred),
            "MAE": mean_absolute_error(true, pred),
            "MSE": mean_squared_error(true, pred),
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
        },
        index=["scores"],
    )
    return scores
df = pd.DataFrame()

uploaded_file = ""

st.markdown("Please contact Bhavani Shankar @ bhavani_5364@hotmail.com for any customization")           
def perform_nlp():
    # Perform time series task
    st.title("Performing NLP task...")
    # Dropdown options
    options = ['Text summarization','Language Translation']
    # Selectbox with manual input
    selected_options = st.selectbox("Select an option or enter a value", options )
    st.write("Selected option:", selected_options)
    
    if selected_options == "Text summarization":
       # Add title on the page
       st.title("Text summarization")

       # Ask user for input text
       input_sent = st.text_area("Input Text", "", height=400)

       ratio = st.slider(
           "Summarization fraction", min_value=0.0, max_value=1.0, value=0.2, step=0.01
       )
       
       # Display named entities
       summarized_text = summarizer.summarize(
           input_sent, ratio=ratio, language="english", split=True, scores=True
       )

       for sentence, score in summarized_text:
           st.write(sentence)
    
           
       
       
    elif selected_options == "Language Translation":
        
        
        Languages = {'afrikaans':'af','albanian':'sq','amharic':'am','arabic':'ar','armenian':'hy','azerbaijani':'az','basque':'eu','belarusian':'be','bengali':'bn','bosnian':'bs','bulgarian':'bg','catalan':'ca','cebuano':'ceb','chichewa':'ny','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','corsican':'co','croatian':'hr','czech':'cs','danish':'da','dutch':'nl','english':'en','esperanto':'eo','estonian':'et','filipino':'tl','finnish':'fi','french':'fr','frisian':'fy','galician':'gl','georgian':'ka','german':'de','greek':'el','gujarati':'gu','haitian creole':'ht','hausa':'ha','hawaiian':'haw','hebrew':'iw','hebrew':'he','hindi':'hi','hmong':'hmn','hungarian':'hu','icelandic':'is','igbo':'ig','indonesian':'id','irish':'ga','italian':'it','japanese':'ja','javanese':'jw','kannada':'kn','kazakh':'kk','khmer':'km','korean':'ko','kurdish (kurmanji)':'ku','kyrgyz':'ky','lao':'lo','latin':'la','latvian':'lv','lithuanian':'lt','luxembourgish':'lb','macedonian':'mk','malagasy':'mg','malay':'ms','malayalam':'ml','maltese':'mt','maori':'mi','marathi':'mr','mongolian':'mn','myanmar (burmese)':'my','nepali':'ne','norwegian':'no','odia':'or','pashto':'ps','persian':'fa','polish':'pl','portuguese':'pt','punjabi':'pa','romanian':'ro','russian':'ru','samoan':'sm','scots gaelic':'gd','serbian':'sr','sesotho':'st','shona':'sn','sindhi':'sd','sinhala':'si','slovak':'sk','slovenian':'sl','somali':'so','spanish':'es','sundanese':'su','swahili':'sw','swedish':'sv','tajik':'tg','tamil':'ta','telugu':'te','thai':'th','turkish':'tr','turkmen':'tk','ukrainian':'uk','urdu':'ur','uyghur':'ug','uzbek':'uz','vietnamese':'vi','welsh':'cy','xhosa':'xh','yiddish':'yi','yoruba':'yo','zulu':'zu'}
        
        
        translator = google_translator()
        st.title("Language Translator:balloon:")
        
        
        text = st.text_area("Enter text:",height=None,max_chars=None,key=None,help="Enter your text here")
        
        option1 = st.selectbox('Input language',
                              ('english', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch',  'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'a', 'telugu','thai', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'))
        
        option2 = st.selectbox('Output language',
                               ('malayalam', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'))
        
        value1 = Languages[option1]
        value2 = Languages[option2]
        
        if st.button('Translate Sentence'):
            if text == "":
                st.warning('Please **enter text** for translation')

            else:
                translate = translator.translate(text, src=value1, dest=value2)
                trans_text = str(translate.text)  # Extract the translation text from the Translation object
                st.write(trans_text)
               # Convert the translated text to speech
                tts = gTTS(text=trans_text, lang=value2)  # value2 is the target language code
                audio_file_path = "translated_audio.mp3"  # Define the path for the audio file
                
                # Save the audio file
                tts.save(audio_file_path)
                
                # Display a link to download the audio file
                st.audio(audio_file_path, format="audio/mp3")
                # st.write("Audio playback:")
                # st.audio(audio_file_path)

def perform_regression():     
        df = st.empty() 
    # Function to load data from CSV file
        try:
            df,uploaded_file = load_data()
            
        except:
            
            pass
        
            
        
        if df is not None:
            # Separate the target variable from features
            dropdown = {
                "Model": None,
                "Forecast": None,
                }

            dropdown_method = st.sidebar.selectbox("Training/Forecast", list(dropdown.keys()))

            
            try:
                if dropdown_method == "Model":
                       
                        variables = df.columns.tolist()
                        y = st.selectbox("Select Target Variable", variables)
                        X = st.multiselect("Select Dependent Variables", variables)
                        y = df[y]
                        X = df[X]
                        col_name = y.name  # Get the column name of the selected target variable
                        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='regression')
                        if st.button('submit'):
                    
                            # automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression')
                            automl = AutoMLSearch(X_train=X_train,y_train=y_train,problem_type='regression',max_batches=1,optimize_thresholds=True)
                            automl.search()
                            
                           
                            y_pred = automl.best_pipeline.predict(X_test)
                           
                            y_pred = round(y_pred,0)
                            scores = calculate_scores(y_test.values,y_pred.values)
                            st.write(scores)
                            y_pred = pd.DataFrame(y_pred)
                            y_pred["Actual"] = y_test
                            y_pred = round(y_pred,0)
                            y_pred = y_pred[["Actual",col_name]]
                            st.dataframe(y_pred)
                            
                            # Get the uploaded file name without extension
                            
                        
                            # Get the uploaded file name without extension
                            file_name_without_extension = uploaded_file.name.rsplit('.', 1)[0]
                            
                            # Save the model to disk with the uploaded file name as suffix
                            model_filename = f'{file_name_without_extension}_finalized_model.sav'
                            joblib.dump(automl.best_pipeline, model_filename)
                            # Create a line graph for y_test vs y_pred
                            # Create a plot with two lines: one for actual values and one for predicted values
                            plt.figure(figsize=(8, 6))
                            
                            
                            plt.plot(y_pred["Actual"].values, label="Actual Values", marker='o')
                            plt.plot(y_pred[col_name].values, label="Predicted Values", marker='x')
                            plt.xlabel("Index")
                            plt.ylabel("Values")
                            plt.title("Actual Values vs Predicted Values")
                            plt.legend()
                            plt.grid()                                              
                            st.pyplot(plt)
            except:
                st.write("Please upload a csv file as an input")
                pass
                        
            if dropdown_method == "Forecast":
               
                try:
                    st.title('Any Domain Analytics')
                    st.header("Forecasting")
                
                    # File Uploader
                    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
                    st.write("uploaded_file")
                    
                    
                    if uploaded_file is not None:
                        df = pd.read_csv(uploaded_file) 
                        st.subheader("Data Preview")
                        st.dataframe(df.head())  # Display the first few rows of the data
                        
                        # Select independent and dependent variables
                        st.subheader("Select Independent and Dependent Variables")
                        variables = df.columns.tolist()
                        selected_target_variable = st.selectbox("Select Target Variable", variables)
                        selected_dependent_variables = st.multiselect("Select Dependent Variables", variables)
                        y = df[selected_target_variable]  # Replace with the actual selected target variable
                        X = df[selected_dependent_variables]  # Replace with the actual selected dependent variables
                        folder_path = "*.sav"  # Replace "path_to_folder" with the actual path to your folder
    
                        # Get a list of all .sav files in the specified folder
                        file_list = glob.glob(folder_path)
    
                        # Load the selected model
                        selected_options = st.selectbox("Select an appropriate saved model or enter a value", file_list)
                        model = joblib.load(selected_options)
                        
                        if st.button('submit'):
                            y_pred_new_Data = model.predict(X)
                            y_pred_new_Data = round(y_pred_new_Data,0)
                            scores = calculate_scores(y.values,y_pred_new_Data.values)
                            st.write(scores)
                           # Create and display dataframe with predictions and actual values
                            y_pred_new_Data = pd.DataFrame({"Actual": y.values, "Predicted": y_pred_new_Data})
                            st.dataframe(y_pred_new_Data)
                        else:
                            st.write("No CSV file uploaded")
                            
                        plt.figure(figsize=(8, 6))
                        
                        
                        plt.plot(y_pred_new_Data["Actual"].values, label="Actual Values", marker='o')
                        plt.plot(y_pred_new_Data["Predicted"].values, label="Predicted Values", marker='x')
                        plt.xlabel("Index")
                        plt.ylabel("Values")
                        plt.title("Actual Values vs Predicted Values")
                        plt.legend()
                        plt.grid()                                              
                        st.pyplot(plt)
                        # Create a download button
                        pred_csv = convert_to_csv(y_pred_new_Data)
                        download_button = st.download_button(label='Download Output Data', data=pred_csv, file_name='pred_csv.csv')
                except:
                    st.write("Please upload a new data csv file to do the forecasting")
                    pass
                

                        
               
        
          
# Function to perform classification task using EvalML
# Function to perform classification task using EvalML
def perform_classification():
    st.title("Performing classification task...")
    st.write("You may insert your classification code here to run and populate the results")
    
    
    df = st.empty() 
    # Function to load data from CSV file
    try:
        df,uploaded_file = load_data()
        
    except:
        st.write("Please upload a csv file as an input")
        pass
        
    if df is not None:
        # Separate the target variable from features
        dropdown = {
            "Model": None,
            "Prediction": None,
            }

        dropdown_method = st.sidebar.selectbox("Training/Prediction", list(dropdown.keys()))
        
    ##########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    try:
        if dropdown_method == "Model":
               
                variables = df.columns.tolist()
                y = st.selectbox("Select Target Variable", variables)
                X = st.multiselect("Select Dependent Variables", variables)
                y = df[y]
                X = df[X]
                col_name = y.name  # Get the column name of the selected target variable
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='multiclass')
                if st.button('submit'):
        
                    # Perform classification using EvalML
                    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass")
                    automl.search()
            
                    # Get the predictions on the test set using the best pipeline
                    y_pred = automl.best_pipeline.predict(X)
                    
                    df_pred = pd.DataFrame(y,columns=['y'])
                    df_pred["y"] = y
                    df_pred["y_pred"] = y_pred
                    st.write("Compare Test data with Predicted data")
                    st.write(df_pred)
                    # Get the uploaded file name without extension
                    file_name_without_extension = uploaded_file.name.rsplit('.', 1)[0]
                    
                    # Save the model to disk with the uploaded file name as suffix
                    model_filename = f'{file_name_without_extension}_finalized_model.sav'
                    joblib.dump(automl.best_pipeline, model_filename)
         
                    # Compute the confusion matrix
                    classes = np.unique(y)  # Extract unique classes from the target variable
                    cm = confusion_matrix(y, y_pred, labels=classes)
         
                    # Display the confusion matrix using a heatmap
                    st.subheader("Confusion Matrix")
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    st.pyplot(plt)  # Pass the Matplotlib figure directly to st.pyplot()
        

    except:
        
        pass
    
    if dropdown_method == "Prediction":
       
        try:
            st.title('Classification Prediction')
            st.header("Prediction")
        
            # File Uploader
            uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
            st.write("uploaded_file")
            
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file) 
                st.subheader("Data Preview")
                st.dataframe(df.head())  # Display the first few rows of the data
                
                # Select independent and dependent variables
                st.subheader("Select Independent and Dependent Variables")
                variables = df.columns.tolist()
                selected_target_variable = st.selectbox("Select Target Variable", variables)
                selected_dependent_variables = st.multiselect("Select Dependent Variables", variables)
                y = df[selected_target_variable]  # Replace with the actual selected target variable
                X = df[selected_dependent_variables]  # Replace with the actual selected dependent variables
                folder_path = "*.sav"  # Replace "path_to_folder" with the actual path to your folder

                # Get a list of all .sav files in the specified folder
                file_list = glob.glob(folder_path)

                # Load the selected model
                selected_options = st.selectbox("Select an appropriate saved model or enter a value", file_list)
                model = joblib.load(str(selected_options))
                
                if st.button('submit'):
                    st.write("Button Clicked")  # Debugging output
                    st.write(model)
                    
                    y_pred_new_Data = model.predict(X)
                    try:
                        y_pred_new_Data = round(y_pred_new_Data, 0)
                    except:
                        pass
                    y_pred_new_Data = pd.DataFrame({"Actual": y.values, "Predicted": y_pred_new_Data})
                    st.dataframe(y_pred_new_Data)
                    try:
                        # Compute the confusion matrix
                        classes = np.unique(y)  # Extract unique classes from the target variable
                        cm = confusion_matrix(y, y_pred_new_Data['Predicted'], labels=classes)
             
                        # Display the confusion matrix using a heatmap
                        st.subheader("Confusion Matrix")
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
                        plt.xlabel("Predicted Label")
                        plt.ylabel("True Label")
                        st.pyplot(plt)  # Pass the Matplotlib figure directly to st.pyplot()
                        # Create a download button
                        pred_csv = convert_to_csv(y_pred_new_Data)
                        download_button = st.download_button(label='Download Output Data', data=pred_csv, file_name='pred_csv.csv')

                    except Exception as e:
                        st.write("An error occurred:", e)
                

                
        except:
            st.write("Please upload an appropriate model file")
            pass
                
    

# Function to perform time series task using EvalML
# Custom preprocessing function for time series data
def preprocess_time_series_data(df, target_column, sequence_length=10):
    # Shift the target column up by the sequence_length
    df['target'] = df[target_column].shift(-sequence_length)
    
    # Drop rows with NaN in the target column (due to the shift)
    df.dropna(subset=['target'], inplace=True)
    
    # Create input sequences and targets for time series regression
    X = []
    y = []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length][target_column].values)
        y.append(df.iloc[i+sequence_length]['target'])
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def perform_time_series_regression():
    st.title("Performing time series regression task...")
    df = st.empty() 
# Function to load data from CSV file
    try:
        df,uploaded_file = load_data()
        
    except:
        
        pass
    
        
    
    if df is not None:
        # Separate the target variable from features
        dropdown = {
            "Model": None,
            "Time Series Forecast": None,
            }

        dropdown_method = st.sidebar.selectbox("Training/Time Series Forecast", list(dropdown.keys()))

        
        try:
            if dropdown_method == "Model":
                   
                    variables = df.columns.tolist()
                    y = st.selectbox("Select Target Variable", variables)
                    X = st.multiselect("Select Dependent Variables", variables)
                    y = df[y]
                    X = df[X]
                    col_name = y.name  # Get the column name of the selected target variable
                    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='regression')
                    if st.button('submit'):
                        try:
                            # automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression')
                            automl = AutoMLSearch(X_train=X_train,y_train=y_train,problem_type='regression',max_batches=1,optimize_thresholds=True)
                            automl.search()
                        except:
                            pass
                       
                        y_pred = automl.best_pipeline.predict(X_test)
                       
                        y_pred = round(y_pred,0)
                        scores = calculate_scores(y_test.values,y_pred.values)
                        st.write(scores)
                        y_pred = pd.DataFrame(y_pred)
                        y_pred["Actual"] = y_test
                        y_pred = round(y_pred,0)
                        # y_pred = y_pred[["Actual",y_pred]]
                        st.dataframe(y_pred)
                        
                        # Get the uploaded file name without extension
                        
                    
                        # Get the uploaded file name without extension
                        file_name_without_extension = uploaded_file.name.rsplit('.', 1)[0]
                        
                        # Save the model to disk with the uploaded file name as suffix
                        model_filename = f'{file_name_without_extension}_finalized_model.sav'
                        joblib.dump(automl.best_pipeline, model_filename)
                        # Create a line graph for y_test vs y_pred
                        # Create a plot with two lines: one for actual values and one for predicted values
                        plt.figure(figsize=(8, 6))
                        
                        
                        plt.plot(y_pred["Actual"].values, label="Actual Values", marker='o')
                        plt.plot(y_pred[col_name].values, label="Predicted Values", marker='x')
                        plt.xlabel("Index")
                        plt.ylabel("Values")
                        plt.title("Actual Values vs Predicted Values")
                        plt.legend()
                        plt.grid()                                              
                        st.pyplot(plt)
        # except Exception as e:
        #     st.write("An error occurred:", e)
        except:
            st.write("Please upload a csv file as an input")
            pass
                    
        if dropdown_method == "Time Series Forecast":
           
            try:
                st.title('Supply Quotient Forecasting')
                st.header("Forecasting")
            
                # File Uploader
                uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
                st.write("uploaded_file")
                
                
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file) 
                    st.subheader("Data Preview")
                    st.dataframe(df.head())  # Display the first few rows of the data
                    
                    # Select independent and dependent variables
                    st.subheader("Select Independent and Dependent Variables")
                    variables = df.columns.tolist()
                    selected_target_variable = st.selectbox("Select Target Variable", variables)
                    selected_dependent_variables = st.multiselect("Select Dependent Variables", variables)
                    y = df[selected_target_variable]  # Replace with the actual selected target variable
                    X = df[selected_dependent_variables]  # Replace with the actual selected dependent variables
                    folder_path = "*.sav"  # Replace "path_to_folder" with the actual path to your folder

                    # Get a list of all .sav files in the specified folder
                    file_list = glob.glob(folder_path)

                    # Load the selected model
                    selected_options = st.selectbox("Select an appropriate saved model or enter a value", file_list)
                    model = joblib.load(selected_options)
                    
                    if st.button('submit'):
                        y_pred_new_Data = model.predict(X)
                        y_pred_new_Data = round(y_pred_new_Data,0)
                        scores = calculate_scores(y.values,y_pred_new_Data.values)
                        st.write(scores)
                       # Create and display dataframe with predictions and actual values
                        y_pred_new_Data = pd.DataFrame({"Actual": y.values, "Predicted": y_pred_new_Data})
                        st.dataframe(y_pred_new_Data)
                    else:
                        st.write("No CSV file uploaded")
                        
                    plt.figure(figsize=(8, 6))
                    
                    
                    plt.plot(y_pred_new_Data["Actual"].values, label="Actual Values", marker='o')
                    plt.plot(y_pred_new_Data["Predicted"].values, label="Predicted Values", marker='x')
                    plt.xlabel("Index")
                    plt.ylabel("Values")
                    plt.title("Actual Values vs Predicted Values")
                    plt.legend()
                    plt.grid()                                              
                    st.pyplot(plt)
                    # Create a download button
                    pred_csv = convert_to_csv(y_pred_new_Data)
                    download_button = st.download_button(label='Download Output Data', data=pred_csv, file_name='pred_csv.csv')

            except:
                st.write("Please upload a new data csv file to do the forecasting")
                pass

        else:
            pass
        
        return      
            
       
def main():
    st.title("Any Domain Analytics")

    # Task selection
    task = st.radio("Select task", ("NLP","Regression", "Classification", "Time Series"))

    if task == "Regression":
        perform_regression()
    elif task == "Classification":
        perform_classification()
    elif task == "Time Series":
        perform_time_series_regression()
    elif task == "NLP":
        perform_nlp()

if __name__ == "__main__":
    main()
