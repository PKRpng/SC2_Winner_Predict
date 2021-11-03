 
import pickle
import streamlit as st
import pandas as pd
import xgboost
#import plotly
from sklearn.preprocessing import OneHotEncoder

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

@st.cache()
# defining the function which will make prediction using the data which the user inputs 
def prediction(player_one, player_two, map_name):
    '''preprocess input, check if input exists, add to a dataframe and predict results'''
    # Pre-processing user input
    player_one, player_two, map_name = player_one.lower(), player_two.lower(), map_name.lower()
    
    if player_one in player_data.players.values:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
        
    if player_two in player_data.players.values:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
 
    if map_name in map_data.maps.values:
        pass
    else:
        st.error('cant find this map, please select another')
     
    df = pd.DataFrame(data={'map_name':[map_name], 'player_one':[player_one], 'player_two':[player_two]})
    
    #encode
    X = enc.transform(df)
    
    # Making predictions 
    prediction = model.predict(X)
     
    if prediction == 0:
        pred = player_two
    else:
        pred = player_one
    return pred

#def get_stats(dataframe):
 #   chart_data = dataframe[dataframe.date, dataframe.map_name]
  #  chart_data = chart_data.groupby('map_name').count()
   # chart_data.reset_index()
    #return chart_data

def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:black;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Predict Starcraft2 Game Winner</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    playerOne = st.text_input('Player One','Serral')
    playerTwo = st.text_input('Player Two','Maru')
    mapName = st.selectbox('Map',map_data.maps)
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(playerOne, playerTwo, mapName) 
        st.success('Predicted winner is {}'.format(result))
    
    st.markdown('Top players: Serral, Maru, Rogue, Dark, Innovation', unsafe_allow_html=False)
    
    #st.bar_chart(get_stats(stats))
    
if __name__=='__main__': 
    main()
