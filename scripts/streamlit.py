{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e1625a1-d8cf-429f-9546-c1df83f984ea",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad77c80a-35e3-4604-8617-b50b206ba502",
   "metadata": {},
   "outputs": [],
   "source": [
    "%%writefile app.py\n",
    " \n",
    "import pickle\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "\n",
    "# loading the trained model\n",
    "pickle_in = open('xgboost_model.pkl', 'rb') \n",
    "model = pickle.load(pickle_in)\n",
    " \n",
    "@st.cache()\n",
    "  \n",
    "players = ['Serral']\n",
    "maps = ['Jagannatha']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bbf9b1a-9048-445e-91fb-ab884745ad16",
   "metadata": {},
   "outputs": [],
   "source": [
    "# defining the function which will make the prediction using the data which the user inputs \n",
    "def prediction(player_one, player_two, map_name):\n",
    "    df = pd.DataFrame(cols = [players, maps])\n",
    "\n",
    "    # Pre-processing user input\n",
    "    player_one, player_two = player_one.lower(), player_two.lower()\n",
    "    \n",
    "    if player_one in players:\n",
    "        df[player_one] = 1\n",
    "    else:\n",
    "        st.error('cant find this player, make sure the name is correct')\n",
    "    if player_two in players:\n",
    "        df[player_two] = 1\n",
    "    else:\n",
    "        st.error('cant find this player, make sure the name is correct')\n",
    " \n",
    "    if map_name in maps:\n",
    "        df[map_name] = 1\n",
    "    else:\n",
    "        st.error('cant find this map, please select another')\n",
    " \n",
    "    # Making predictions \n",
    "    prediction = model.predict(df)\n",
    "     \n",
    "    if prediction == 0:\n",
    "        pred = player_two\n",
    "    else:\n",
    "        pred = player_one\n",
    "    return pred\n",
    "      \n",
    "def main():       \n",
    "    # front end elements of the web page \n",
    "    html_temp = \"\"\" \n",
    "    <div style =\"background-color:black;padding:13px\"> \n",
    "    <h1 style =\"color:white;text-align:center;\">SC2 Predict Game Winner</h1> \n",
    "    </div> \n",
    "    \"\"\"\n",
    "      \n",
    "    # display the front end aspect\n",
    "    st.markdown(html_temp, unsafe_allow_html = True) \n",
    "      \n",
    "    # following lines create boxes in which user can enter data required to make prediction \n",
    "    playerOne = st.text_input('Player One','Byun')\n",
    "    playerTwo = st.text_input('Player Two','Maru')\n",
    "    mapName = st.selectbox('Map',('Jagannatha', 'Romanticide'))\n",
    "    result =\"\"\n",
    "      \n",
    "    # when 'Predict' is clicked, make the prediction and store it \n",
    "    if st.button(\"Predict\"): \n",
    "        result = prediction(playerOne, playerTwo, mapName) \n",
    "        st.success('Predicted winner is {}'.format(result))\n",
    "\n",
    "if __name__=='__main__': \n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
