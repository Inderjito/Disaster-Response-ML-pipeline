## Disaster Response Pipeline Project
----------------------------------------
For this project labeled tweets and messages from real life disaster was provided by Figure Eight. The task was to prepare a ETL pipeline using ML pipeline to build a supervised Learning model that will process and categorize data from CSV files and then load into a SQLlite database. The web app will extract data  from database and classified new messages for 36 catoegories. ML is critical for organization to understand which messages are relevant and which to priortize during a disaster.

### Data
The data contains 26,248 labeled messages that were sent during past disasters around the world. Each message labeled as 1 or more of the following 36 categories:

'related', 'request', 'offer', 'aid_related', 
'medical_help', 'medical_products',
'search_and_rescue', 'security', 'military', 
'child_alone', 'water', 'food', 'shelter', 
'clothing', 'money', 'missing_people', 'refugees', 
'death', 'other_aid', 'infrastructure_related', 
'transport', 'buildings', 'electricity', 'tools', 
'hospitals', 'shops', 'aid_centers', 
'other_infrastructure', 'weather_related', 
'floods', 'storm', 'fire', 'earthquake', 'cold', 
'other_weather', 'direct_report'

#### Project Components
1.  **ETL Pipeline**
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. **ML Pipeline**
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3.**Flask Web App**
Run command to run web app: python run.py

.**Instructions to run**
Run the following commands in the project's root directory to set up your database and model.

-To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
-To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/ or you can change url in code.

#### Interface Homepage 
![image](https://drive.google.com/file/d/1guOtJhtYjrmoDHn7L_C8g6Jalb4xadn-/view?usp=sharing)




