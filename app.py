 
import pickle
import streamlit as st
import pandas as pd
import xgboost

# loading the trained model
pickle_in = open('models/xgboost_model.pkl', 'rb') 
model = pickle.load(pickle_in)

pickle_in = open('models/enc.pkl', 'rb') 
enc = pickle.load(pickle_in)

 

@st.cache()
# defining the function which will make prediction using the data which the user inputs 
def prediction(player_one, player_two, map_name):
    '''preprocess input, check if input exists, add to a dataframe and predict results'''
    # Pre-processing user input
    player_one, player_two = player_one.lower(), player_two.lower()
    
    if player_one in player_data.players:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
        
    if player_two in player_data.players:
        pass
    else:
        st.error('cant find this player, make sure the name is correct')
 
    if map_name in map_data.maps:
        pass
    else:
        st.error('cant find this map, please select another')
     
    df = pd.DataFrame(data={'map_name':map_name, 'player_one':player_one, 'player_two':player_two})
    
    #encode
    X = enc.transform(df)
    
    # Making predictions 
    prediction = model.predict(X)
     
    if prediction == 0:
        pred = player_two
    else:
        pred = player_one
    return pred
      
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:black;padding:13px"> 
    <h1 style ="color:white;text-align:center;">SC2 Predict Game Winner</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    playerOne = st.text_input('Player One','Byun')
    playerTwo = st.text_input('Player Two','Maru')
    mapName = st.selectbox('Map',map_data.maps)
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(playerOne, playerTwo, mapName) 
        st.success('Predicted winner is {}'.format(result))

if __name__=='__main__': 
    main()
