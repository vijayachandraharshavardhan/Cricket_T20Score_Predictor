from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignore warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams and cities
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
    'Christchurch', 'Trinidad'
]

# Root redirects to intro
@app.route('/')
def root():
    return redirect('/intro')

# Intro page
@app.route('/intro')
def intro():
    return render_template('intro.html')

# Predictor page
@app.route('/predict', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None

    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        current_score = int(request.form['current_score'])
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets'])
        last_five = int(request.form['last_five'])

        # Validation
        if batting_team == bowling_team:
            error_message = "Batting and Bowling teams must be different."
        elif not (5 < overs < 20):
            error_message = "Overs must be greater than 5 and less than 20."
        elif wickets > 10:
            error_message = "Wickets cannot be more than 10."
        elif last_five > current_score:
            error_message = "Runs in last 5 overs cannot exceed the current score."
        else:
            balls_left = 120 - int(overs * 6)
            wickets_left = 10 - wickets
            crr = current_score / overs if overs > 0 else 0

            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'current_score': [current_score],
                'balls_left': [balls_left],
                'wicket_left': [wickets_left],
                'current_run_rate': [crr],
                'last_five': [last_five]
            })

            result = pipe.predict(input_df)
            prediction = int(result[0])

    return render_template(
        'index.html',
        teams=sorted(teams),
        cities=sorted(cities),
        prediction=prediction,
        error_message=error_message
    )

if __name__ == '__main__':
    app.run(debug=True)
