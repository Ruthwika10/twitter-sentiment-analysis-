# Twitter Sentiment Analysis

This project analyzes the sentiment of tweets from a CSV dataset using Python and visualizes the results.

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/chidupudi/twitter-sentiment-with-csv.git
cd twitter-sentiment-with-csv


### 2. Create a Virtual Environment
python -m venv venv

Activate the environment:
- On Windows:
venv\Scripts\activate

- On macOS/Linux:
source venv/bin/activate


### 3. Install Requirements
pip install -r requirements.txt


### 4. Run the Analysis
python main.py

Follow the prompts to enter a keyword and number of tweets to analyze. The script will display sentiment statistics and graphs.

## Output
- Sentiment statistics and visualizations
- Optionally, results can be saved to `sentiment_analysis_results.json`

## Requirements
- Python 3.7+
- See `requirements.txt` for dependencies

## Files
- `main.py`: Main analysis script
- `twitter_dataset.csv`: Dataset of tweets
- `requirements.txt`: Python dependencies
- `sentiment_analysis_results.json`: (Optional) Output file for results

## License
See `License.txt` for license information.
