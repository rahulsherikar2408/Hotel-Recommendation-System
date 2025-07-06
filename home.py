import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from math import sin, cos, sqrt, atan2, radians
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


st.set_page_config(page_title="Hotel Recommendation System", layout="wide")
secret_value_0 = '4519d076b432f5'
R = 6373.0  # Earth's Radius in kilometers
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
sw = stopwords.words('english')
lemm = WordNetLemmatizer()

room_no=[
     ('king',2),
   ('queen',2), 
    ('triple',3),
    ('master',3),
   ('family',4),
   ('murphy',2),
   ('quad',4),
   ('double-double',4),
   ('mini',2),
   ('studio',1),
    ('junior',2),
   ('apartment',4),
    ('double',2),
   ('twin',2),
   ('double-twin',4),
   ('single',1),
     ('diabled',1),
   ('accessible',1),
    ('suite',2),
    ('one',2)
   ]

def load_data():
    global hotel_details
    global hotel_rooms
    global hotel_cost
    hotel_details=pd.read_csv('Hotel_details.csv', encoding='latin1')
    hotel_rooms=pd.read_csv('Hotel_Room_attributes.csv', encoding='latin1')
    hotel_cost=pd.read_csv('hotels_RoomPrice.csv', encoding='latin1')
    del hotel_details['id']
    del hotel_rooms['id']
    del hotel_details['zipcode']

    hotel_details=hotel_details.dropna()
    hotel_rooms=hotel_rooms.dropna()
    hotel_details.drop_duplicates(subset='hotelid',keep=False,inplace=True)
    global hotel
    hotel=pd.merge(hotel_rooms,hotel_details,left_on='hotelcode',right_on='hotelid',how='inner')
    hotel['roomamenities']=hotel['roomamenities'].str.replace(': ;',', ')
    del hotel['hotelid']
    del hotel['curr']
    del hotel['Source']
    calc()
    global url
    url = "https://us1.locationiq.com/v1/search.php"

    columns_to_drop = ['id', 'refid', 'websitecode', 'dtcollected', 'ratedate', 'los', 'guests', 'roomtype', 'netrate', 
                   'ratedescription', 'ratetype', 'sourceurl', 'roomamenities', 'ispromo', 'closed', 'discount', 
                   'promoname', 'status_code', 'taxstatus', 'taxtype', 'taxamount', 'proxyused', 'israteperstay', 
                   'hotelblock', 'input_dtcollected']

# Drop only columns that exist in the DataFrame
    columns_in_df = hotel_cost.columns.intersection(columns_to_drop)
    hotel_cost = hotel_cost.drop(columns_in_df, axis=1)

#To reccomend we are gonna check how much does the price vary from room to room if 
#the varience is small enough then it is better for them to recommend the hotel
    hot=hotel_cost.groupby(['hotelcode','maxoccupancy'])

    hotel_cost.sort_values(by=['onsiterate'],ascending=False)
    hotel_cost=hotel_cost.drop_duplicates(subset=['hotelcode','maxoccupancy'],keep='first')

    var = hot['onsiterate'].var().to_frame('varience')
    l = []

    for i in range(hotel_cost.shape[0]):
        # Access the first level of the index
        var1 = var[var.index.get_level_values(0) == hotel_cost.iloc[i, 0]]
    
    # Check if var1 is not empty
        if not var1.empty:
            # Access the second level of the index
            var2 = var1[var1.index.get_level_values(1) == hotel_cost.iloc[i, 3]]
        
            # Ensure var2 is not empty and has the expected data
            if not var2.empty:
                # Append the variance value
                l.append(var2['varience'].iloc[0])
            else:
                l.append(np.nan)  # Or another default value
        else:
            l.append(np.nan)  # Or another default value


    hotel_cost['var']=l
    hotel_cost=hotel_cost.fillna(0)
    hotel_cost['mealinclusiontype']=hotel_cost['mealinclusiontype'].replace(0,'No Complimentary')
    hotel1=pd.merge(hotel,hotel_cost,left_on=['hotelcode','guests_no'],right_on=['hotelcode','maxoccupancy'],how='inner')
    hotel1=hotel1.drop_duplicates(subset=['hotelcode','maxoccupancy'],keep='first')
    global optimum_band
    optimum_band=pd.read_csv('hotel_price_min_max - Formula.csv')



def calc():
    guests_no=[]
    for i in range(hotel.shape[0]):
        temp=hotel['roomtype'][i].lower().split()
        flag=0
        for j in range(len(temp)):
            for k in range(len(room_no)):
                if temp[j]==room_no[k][0]:
                    guests_no.append(room_no[k][1])
                    flag=1
                    break
            if flag==1:
                break
        if flag==0:
            guests_no.append(2)
    hotel['guests_no']=guests_no


def hybrid(address, city, number, rating, room_type, description, features):
    features = features.lower()
    room_type = room_type.lower()
    description = description.lower()

    # Tokenize and lemmatize the inputs
    features_tokens = word_tokenize(features)
    description_tokens = word_tokenize(description)

    f1_set = {w for w in features_tokens if not w in sw}
    d1_set = {w for w in description_tokens if not w in sw}

    f_set = set(lemm.lemmatize(se) for se in f1_set)
    d_set = set(lemm.lemmatize(se) for se in d1_set)

    data = {
        'key': secret_value_0,
        'q': address,
        'format': 'json'
    }
    response = requests.get(url, params=data)
    dist = []
    lat1, long1 = response.json()[0]['lat'], response.json()[0]['lon']
    lat1 = radians(float(lat1))
    long1 = radians(float(long1))

    hybridbase = hotel
    hybridbase['city'] = hybridbase['city'].str.lower()
    hybridbase = hybridbase[hybridbase['city'] == city.lower()]
    hybridbase.drop_duplicates(subset='hotelcode', inplace=True, keep='first')
    hybridbase = hybridbase[hybridbase['starrating'] >= rating]  # Filter by rating
    hybridbase = hybridbase.set_index(np.arange(hybridbase.shape[0]))

    for i in range(hybridbase.shape[0]):
        lat2 = radians(hybridbase['latitude'][i])
        long2 = radians(hybridbase['longitude'][i])
        dlon = long2 - long1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        # a = min(1, max(0, a))
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        dist.append(distance)

    hybridbase['distance'] = dist
    hybridbase = hybridbase[hybridbase['distance'] <= 5]
    hybridbase = hybridbase.set_index(np.arange(hybridbase.shape[0]))

    coss = []
    for i in range(hybridbase.shape[0]):
        temp_tokens = word_tokenize(hybridbase['roomamenities'][i])
        temp1_set = {w for w in temp_tokens if not w in sw}
        temp_set = set(lemm.lemmatize(se) for se in temp1_set)
        
        # Calculate similarity with features and description
        rvector = temp_set.intersection(f_set.union(d_set))
        similarity = len(rvector)

        # Check if room_type matches completely
        if room_type in hybridbase['roomtype'][i].lower():
            similarity += 1  # Increase similarity if room_type matches

        coss.append(similarity)

    hybridbase['similarity'] = coss
    return hybridbase.sort_values(by='similarity', ascending=False)


def price_band_based(address,city,number,rating,room_type,description,features):
    h=hybrid(address,city,number,rating,room_type,description,features)
    price_band=pd.merge(h,optimum_band,left_on=['hotelcode'],right_on=['hotelcode'],how='inner')
    price_band=pd.merge(price_band,hotel_cost,left_on=['hotelcode'],right_on=['hotelcode'],how='inner')
    del price_band['min']
    del price_band['max']
    del price_band['hotelcode']
    del price_band['onsiterate']
    del price_band['ratedescription']
    del price_band['distance']
    del price_band['Diff_Min']
    del price_band['Diff_Max']
    del price_band['currency']
    del price_band['propertytype']
    # del price_band['starrating']
    del price_band['latitude']
    del price_band['longitude']
    del price_band['guests_no']
    del price_band['var']
    price_band=price_band[price_band['Score']<=0.5]
    price_band=price_band[price_band['maxoccupancy']>=number]
    return price_band

# User input fields

st.title("🏨 Hotel Recommendation System")
st.subheader("Find the best hotels based on your preferences")

# Section: Input form
st.markdown("### Please provide your preferences:")

# Create columns for better organization
col1, col2 = st.columns(2)

with col1:
    address = st.text_input("📍 Enter the address of your destination")
    city = st.text_input("🏙️ Enter the city")
    number = st.number_input("👥 Number of guests", min_value=1, step=1)
    room_type = st.text_input("🛏️ Room Type (e.g., single, double)")

with col2:
    rating = st.slider("⭐ Minimum hotel rating", min_value=0, max_value=5, step=1)
    description = st.text_area("📄 Room Description (e.g., Non-smoking, City view)")
    features = st.text_area("🛠️ Room Amenities (e.g., WiFi, coffee maker, air-conditioning)")

# Button for getting recommendations
st.markdown(" ")
if st.button("🔍 Get Recommendations"):
    if address and city and room_type and features and number:
        load_data()  # Load the dataset
        recommendations = price_band_based(address, city, number, rating, room_type, description, features).head(5)
        
        if not recommendations.empty:
            st.write("## 🏨 Top Hotel Recommendations:")
            st.dataframe(recommendations[['hotelname','address','city','country','starrating','roomtype','roomamenities','similarity','mealinclusiontype']])  # Show the most relevant columns
            j = 1
            # Display clickable 'Book Now' links

            Hoteln = []
            Url = []
            for i, row in recommendations.iterrows():
                Hoteln.append(row['hotelname'])
                Url.append(row['url'])

            recommendationslink= pd.DataFrame({
                'hotelname': Hoteln,
                'url': Url
            })
            
            st.markdown("""
                <style>
                    .booknow-link {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        position: relative;
                        font-size: 18px;
                        color: white;
                        text-decoration: none;
                    }
                    .booknow-link p {
                        color: white;
                        padding : 8px;
                        margin : 0;
                    }
                    .booknow-link a {
                        text-decoration: none;
                        padding-left: 8px;
                        color: #007bff;
                    }
                    .booknow-link:hover a {
                        color: #0056b3;
                    }
                    .booknow-link:hover .plane {
                        visibility: visible;
                        animation: movePlane 0.5s linear forwards;
                    }
                    .plane {
                        visibility: hidden;
                        position: relative;
                        width: 30px;
                        height: 30px;
                        background-image: url('https://cdn1.iconfinder.com/data/icons/leto-travel-vacation/64/__airport_plane_travel-512.png');
                        background-size: cover;
                    }
                    @keyframes movePlane {
                        from { left: 0; }
                        to { left: 100px; top: 0px; }
                    }
                </style>
            """, unsafe_allow_html=True)
            st.write("### 📲 Book Now Links:")

            final_output = pd.DataFrame(columns=['hotelname', 'url'])
            # Display hotel names with animated 'Book Now' hyperlinks and append to DataFrame
            for index, row in recommendationslink.iterrows():
            # Show hotel name and Book Now hyperlink with animation
                st.markdown(f"""
                    <div class="booknow-link">
                        <div><p>{index+1}. {row['hotelname']}</p></div> 
                        <div><a href="{row['url']}" target="_blank">Book Now</a></div>
                        <div class="plane"></div>
                    </div>
                """, unsafe_allow_html=True)

                new_row = pd.DataFrame({'hotelname': [row['hotelname']], 'url': [row['url']]})
                final_output = pd.concat([final_output, new_row], ignore_index=True)

            # Display the final output DataFrame below the recommendations

        else:
            st.error("No recommendations found based on the provided criteria.")
    else:
        st.warning("Please fill in all required fields.")


