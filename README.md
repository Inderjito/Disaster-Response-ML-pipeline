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

***FILE STRUCTURE***




<img width="783" alt="Screen Shot 2020-12-31 at 12 52 37 PM" src="https://user-images.githubusercontent.com/71035452/103420794-2e928100-4b67-11eb-90c9-8d6d9fafa55a.png">


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

 To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/-


<img width="1337" alt="Screen Shot 2020-12-31 at 12 39 42 PM" src="https://user-images.githubusercontent.com/71035452/103420571-e9218400-4b65-11eb-9061-42435fbd6103.png">


<img width="1307" alt="Screen Shot 2020-12-31 at 12 38 46 PM" src="https://user-images.githubusercontent.com/71035452/103420703-add38500-4b66-11eb-997f-8f1a73f0fc18.png">



