 
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

player_data = pd.DataFrame(data={'players':['agoelazer','alive','armani','astrea','bly','bunny','byun','cham','classic',
                                            'clem','cure','cyan','dark','dayshi','dear','denver','disk','dns','elazer','enderr','epic','erik',
                                            'future','gerald','goblin','gumiho','gungfubanda','guru','harstem','has','hateme','hellraiser','hero','heromarine','hurricane','igmacsed',
                                            'innovation','jimrising','jonsnow','kas','kelazhur','krystianer','lambo','lilbow','mana',
                                            'maru','masa','maxpax','mcanning','miszu','namshar','neeb','nerchio','nice','nightend','nina','optimus',
                                            'parting','patience','pilipili','probe','ptitdrogo','puck','rail','rex','reynor','risky','rogue',
                                            'scarlett','seither','semper','serral','shadown','showtime','silky','skillous','snute','solar','soo',
                                            'sortof','sos','soul','souleer','special','state','stats','stephano','teebul','thezerglord','time',
                                            'tlo','trap','true','tyty','uthermal','vanya','vindicta','zanster','zest','ziggy']}) 
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


race_dict = {'agoelazer':'zerg','alive':'terran','armani':'zerg','astrea':'protoss','bly':'zerg',
             'bunny':'terran','byun':'terran','cham':'zerg','classic':'protoss',
            'clem':'terran','cure':'terran','cyan':'protoss','dark':'zerg','dayshi':'terran','dear':'protoss','denver':'zerg',
             'disk':'protoss','dns':'protoss','elazer':'zerg','enderr':'zerg','epic':'terran','erik':'zerg', 
             'future':'terran','gerald':'protoss','goblin':'protoss','gumiho':'terran','gungfubanda':'protoss','guru':'zerg','harstem':'protoss',
             'has':'protoss','hateme':'zerg','hellraiser':'protoss','hero':'protoss','heromarine':'terran','hurricane':'protoss',
             'igmacsed':'protoss','innovation':'terran','jimrising':'zerg','jonsnow':'zerg','kas':'terran',
             'kelazhur':'terran','krystianer':'protoss','lambo':'zerg','lilbow':'protoss','mana':'protoss',
             'maru':'terran','masa':'terran','maxpax':'protoss','mcanning':'protoss','miszu':'terran','namshar':'zerg','neeb':'protoss',
             'nerchio':'zerg','nice':'protoss','nightend':'protoss','nina':'protoss','optimus':'terran',
             'parting':'protoss','patience':'protoss','pilipili':'protoss','probe':'protoss','ptitdrogo':'protoss',
             'puck':'protoss','rail':'protoss','rex':'zerg','reynor':'zerg','risky':'zerg','rogue':'zerg',
             'scarlett':'zerg','seither':'terran','semper':'terran','serral':'zerg','shadown':'protoss','showtime':'protoss',
             'silky':'zerg','skillous':'protoss','snute':'zerg','solar':'zerg','soo':'zerg',
             'sortof':'zerg','sos':'protoss','soul':'terran','souleer':'zerg','special':'terran','state':'protoss',
             'stats':'protoss','stephano':'zerg','teebul':'protoss','thezerglord':'zerg','time':'terran',
             'tlo':'zerg','trap':'protoss','true':'zerg','tyty':'terran','uthermal':'terran','vanya':'zerg','vindicta':'terran',
             'zanster':'zerg','zest':'protoss','ziggy':'terran'}

stats = pd.read_csv('data/prepared_data.csv')
player_names = player_data.players.values.tolist()
player_names = ' '.join(player_names)

@st.cache()
# defining the function which will make prediction using the data which the user inputs 
def prediction(player_one, player_two, map1, map2, map3):
    '''preprocess input, check if input exists, add to a dataframe and predict results'''
    # Pre-processing user input
    player_one, player_two = player_one.lower(), player_two.lower()
    
    if player_one in player_data.players.values:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
        
    if player_two in player_data.players.values:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
    
    p1_race = race_dict[player_one]
    p2_race = race_dict[player_two]
    
    df = pd.DataFrame(data={'map_name':[map1, map2, map3], 
                            'player_one_race':[p1_race, p1_race, p1_race], 
                            'player_two_race':[p2_race, p2_race, p2_race], 
                            'player_one':[player_one, player_one, player_one], 
                            'player_two':[player_two, player_two, player_two]})
    
    #encode
    X = enc.transform(df)
    
    # Making predictions 
    prediction = model.predict(X)
    
    p1, p2 = 0, 0
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
    
    def change_m1():
        st.session_state.map1Name = map_one
    
    def change_m2():
        st.session_state.map2Name = map_two
    
    def change_m3():
        st.session_state.map3Name = map_three
    
    if 'map1Name' not in st.session_state:
        st.session_state.map1Name = ''
    
    if 'map2Name' not in st.session_state:
        st.session_state.map2Name = ''
        
    if 'map3Name' not in st.session_state:
        st.session_state.map3Name = ''
    
    map_one = st.selectbox('Map 1',map_data.maps, 54, on_change=change_m1)
    map_two = st.selectbox('Map 2',map_data.maps, 2, on_change=change_m2)
    map_three = st.selectbox('Map 3',map_data.maps, 20, on_change=change_m3)

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(playerOne, playerTwo, st.session_state.map1Name, st.session_state.map2Name, st.session_state.map3Name) 
        st.success(result)
    
if __name__=='__main__': 
    main()
