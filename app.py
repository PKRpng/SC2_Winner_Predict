 
import pickle
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy
import plotly
from sklearn.preprocessing import OneHotEncoder
import random

# loading the trained model
pickle_in = open('models/xgboost_model.pkl', 'rb') 
model = pickle.load(pickle_in)

pickle_in = open('models/enc.pkl', 'rb') 
enc = pickle.load(pickle_in)

player_data = pd.DataFrame(data={'players':['serral', 'trap', 'parting','reynor','jimrising','gerald','hellraiser','special',
                                           'uthermal','mana','maxpax','solar','showtime','liquidclem','stats','innovation',
                                           'acerbly','rogue','byun','heromarine','bunny','uwuthermal','lambo','blyonfire',
                                           'zest','vibe','agoelazer','maru','dark','jason','marinelord','clem','tlharstem','neeb',
                                           'neeblet','liquidmana','has','nice','rex','enderr','astrea','probe','teebul','namshar','scarlett',
                                           'nina','future','masa','tlo','epic','skillous','mcanning','pilipili','jonsnow','disk',
                                           'vindicta','cham','kelazhur','erik','thezerglord','denver','geralt','gungfubanda',
                                           'rail','soul','ptitdrogo','dns','vanya','souleer','gostephano','kas','krystianer',
                                           'goblin','ziggy','shadown','time','igmacsed','tyty','llllllllllll','armani','patience',
                                           'sortof','risky','liquidtlo','elazer','lilbow','silky','seither','harstem','dpgparting',
                                           'bly','stephano','soo','dear','hurricane','cyan','liquidthermy','semper','tsgsolar','xkawaiian',
                                           'railgan','beastyqt','zanster','sos','hateme','cure','puck','hero','goreynor','pig',
                                           'liilllliilll','classic','iiiiiiiiiiii','liquidsnute','nerchio','snute','alive','true',
                                           'guru','gumiho','optimus','nightend','dayshi','pokebunny','miszu','state','bop','dpgcure',
                                           'a.i.','drg','arctur','art','dream','bratok','lllllllllll','impact','inzane','rob',
                                           'liquidtaeja','ogzest','iasonu','cloudy','demuslim','firecake']}) 
map_data = pd.DataFrame(data={'maps':['jagannatha le', '2000 atmospheres le', 'oxide le','lightshade le', 'blackburn le', 'romanticide le',
                                        'beckett industries le', 'pillars of gold le', 'submarine le','deathaura le', 'ruins of seras', 
                                        'ice and chrome le','nightshade le', 'lerilak crest', 'world of sleepers le',
                                        'ever dream le', 'eternal empire le', 'golden wall le','ephemeron le', 'simulacrum le', 
                                        'zen le', 'dreamcatcher le','acid plant le', 'triton le', 'disco bloodbath le', 'acropolis le',
                                        'thunderbird le', "winter's gate le", 'cyber forest le',"king's cove le", 'kairos junction le', 
                                        'new repugnancy le','battle on the boardwalk le', 'port aleksander le', 'year zero le',
                                        'automaton le', 'fort alexander le', 'para site le','echo le (void)', 'galactic process le', 
                                        'lost and found le','mech depot le', 'new gettysburg le', 'odyssey le', 'backwater le',
                                        'blueshift le', 'daybreak le (void)', 'dusk towers','cerulean fall le', 'ascension to aiur le', 
                                        'catalyst le','16-bit le', 'blackpink le', 'neon violet square le','eastwatch le', 'abyssal reef le', 
                                        'cactus valley le','sequencer le', 'proxima station le', 'newkirk precinct te (void)',
                                        "bel'shir vestige le (void)", 'honorgrounds le','overgrowth le (void)', 'habitation station le (void)',
                                        'frozen temple', 'king sejong station le', 'frost le','apotheosis le', 'ruins of endion', 
                                        'prion terraces', 'ulrena','orbital shipyard', 'central protocol', 'terraform le (void)','coda le (void)']})  

stats = pd.read_csv('data/prepared_data.csv')
player_names = player_data.players.values.tolist()
player_names = ' '.join(player_names)

random_map = stats.groupby('map_name').count().sort_values(by='date', ascending=False).iloc[:15].reset_index().map_name.values.tolist()

@st.cache()
# defining the function which will make prediction using the data which the user inputs 
def prediction(player_one, player_two, map1, map2, map3):
    '''preprocess input, check if input exists, add to a dataframe and predict results'''
    # Pre-processing user input
    player_one, player_two = player_one.lower(), player_two.lower(), map_name.lower()
    
    if player_one in player_data.players.values:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
        
    if player_two in player_data.players.values:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
    
    df = pd.DataFrame(data={'map_name':[map1, map2, map3], 'player_one':[player_one, player_one, player_one], 'player_two':[player_two, player_two, player_two]})
    
    #encode
    X = enc.transform(df)
    
    # Making predictions 
    prediction = model.predict(X)
    
    if prediction[0] == 0:
        p1 = 1
    else:
        p2 = 1
    
    if prediction[1] == 0:
        p1 += 1
    else:
        p2 += 1
        
    if prediction[2] == 0:
        p1 += 1
    else:
        p2 += 1
    
    if p1 - p2 > 0:
        score = f'{player_one} is predicted to win the match with a {p1} to {p2} score'
    else: 
        score = f'{player_two} is predicted to win the match with a {p2} to {p1} score'
    
    return score

def rand_map():

    random_list = random.sample(random_map, 3)
    m1_index = map_data[map_data.maps == random_list[0]].index.values.tolist()[0]
    m2_index = map_data[map_data.maps == random_list[1]].index.values.tolist()[0]
    m3_index = map_data[map_data.maps == random_list[2]].index.values.tolist()[0]
     
    map1Name = st.selectbox('Map 1',map_data.maps, m1_index)
    map2Name = st.selectbox('Map 2',map_data.maps, m2_index)
    map3Name = st.selectbox('Map 3',map_data.maps, m3_index)  
        
    return map1Name, map2Name, map3Name
    
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:black;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Predict Starcraft 2 Match Winner</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    playerOne = st.text_input('Player One','Serral')
    playerTwo = st.text_input('Player Two','Maru')
    
    st.markdown('Top players: Serral, Maru, Rogue, Dark, Innovation', unsafe_allow_html=False)
    
    with st.expander("See all player names"):
        names = player_data.players.values.tolist
        st.write(player_names)
     
    st.write('Select random maps from top 15 most frequently played maps')
    #Fill maps with random map from top 10
    if st.button("Random Maps"):
        map1Name, map2Name, map3Name = rand_map()

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(playerOne, playerTwo, map1Name, map2Name, map3Name) 
        st.success(result)
    
if __name__=='__main__': 
    main()
