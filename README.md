# Hotel Recommendation System

This project is a hybrid recommendation system that helps users find the best hotels based on their preferences such as location, room type, rating, and amenities. The system utilizes a dataset from Kaggle to provide accurate recommendations using a combination of user inputs and machine learning techniques.

## Features
- 📍 **Location-based Recommendations**: Users can input the city and destination address to receive hotel recommendations in that area.
- ⭐ **Rating-based Filtering**: The system allows users to filter hotels based on a minimum star rating.
- 🏨 **Room Type & Amenities Selection**: Users can specify their desired room type and amenities, such as air conditioning, free WiFi, and more.
- 🌐 **Interactive UI**: Built using Streamlit for an easy-to-use interface.
- 🔗 **Booking Links**: Each recommendation comes with a "Book Now" link for user convenience.

## Dataset
The project uses a hotel recommendation dataset from Kaggle.

Download the dataset through this [link](https://www.kaggle.com/api/v1/datasets/download/keshavramaiah/hotel-recommendation)

## Tech Stack
- **Python**: Core programming language used for development.
- **Streamlit**: Used for building the user interface.
- **Scikit-learn**: Applied for machine learning algorithms to improve recommendation accuracy.
- **NLTK**: Used for text processing and similarity analysis.

## LocationIQ API

### Overview
The Hotel Recommendation System integrates **LocationIQ API** to enhance location-based features, allowing users to input their desired destination and city. The API provides accurate geocoding, which is essential for converting addresses or place names into geographic coordinates, helping to identify the location of hotels relative to the user's preferences.

### Features of LocationIQ API:
- **Geocoding**: Converts addresses and place names into geographical coordinates (latitude and longitude).
- **Reverse Geocoding**: Converts geographic coordinates into human-readable addresses.
- **Place Search**: Returns information on nearby points of interest based on geographic coordinates.

### Why LocationIQ?
LocationIQ was chosen for its simplicity, affordability, and accuracy in providing geographic data, which is essential for hotel recommendations based on user input.

### Setup
To use the LocationIQ API in your project, follow these steps:

1. **Create an Account**: Sign up at [LocationIQ](https://locationiq.com/) and generate an API key.
2. **Install Required Libraries**:
   - Install the `requests` library for making HTTP requests to the LocationIQ API:
     
     ```bash
     pip install requests
     ```
3. **Integrate API in the Project**:
   Use the API to fetch location data:
   ```python
   import requests

   def get_location(address, api_key):
       url = f"https://us1.locationiq.com/v1/search.php?key={api_key}&q={address}&format=json"
       response = requests.get(url)
       location_data = response.json()
       return location_data[0]['lat'], location_data[0]['lon']


## Installation

To set up the project, first clone the repository:

```bash
git clone https://github.com/yourusername/hotel-recommendation-system.git
```
Then, navigate into the project directory:

```bash
cd hotel-recommendation-system
```

Install the necessary Python libraries through your terminal:

```bash
pip install numpy pandas streamlit nltk requests scikit-learn
```
or
```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, use the following command:

```bash
streamlit run home.py
```

So, when you run the project locally, you can access it via:

```bash
http://localhost:8501
```

## How It Works

1. **Input Preferences**: Users provide information such as city, destination, number of guests, room type, rating, and desired amenities.
2. **Recommendation Engine**: The system processes the inputs and returns a list of hotels that match the criteria.
3. **Book Now Links**: Each recommendation comes with a direct booking link to streamline the reservation process.

## Contribution

Feel free to submit issues or fork the repository to contribute.
