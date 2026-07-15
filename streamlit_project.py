import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from mplsoccer import VerticalPitch
import joblib
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import xgboost
import shap
from streamlit_option_menu import option_menu
from scipy.special import expit
from st_clickable_images import clickable_images
import base64
# from streamlit_image_select import image_select
import os
import io
from PIL import Image, ImageDraw
import plotly.graph_objects as go


shotsMultiplier = 500
lastRound = 18
pd.options.mode.chained_assignment = None
# st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""
    <style>
    .card {
        background-color:rgb(0, 8, 255); /* Colore sfondo card */
        border: 2px solidrgb(239, 239, 239); /* Colore bordo */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        max-width: 300px;
        margin: auto;
        font-family: Arial, sans-serif;
        text-align: center;
        height: 90%;
    }
    .card img {
        border-radius: 10px;
        width: 100%;
        height: auto;
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 15px;
        color: rgb(255, 255, 255);
    }
    .card-row {
        display: flex;
        justify-content: space-between;
        font-size: 1rem;
        margin: 10px 0;
        color: rgb(255, 255, 255);
    }
    .card-difference {
        font-size: 1.2rem;
        font-weight: bold;
        color:rgb(255, 255, 255); /* Rosso */
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

def image_to_uri(path, selected=False):
            img = Image.open(path).convert("RGBA")
            border = 6

            # sfondo sempre trasparente, cambia solo lo spazio per il contorno
            canvas = Image.new("RGBA", (img.width + border * 2, img.height + border * 2), (0, 0, 0, 0))
            canvas.paste(img, (border, border), img)

            if selected:
                draw = ImageDraw.Draw(canvas)
                outline_color = (0, 150, 255, 255)  # blu/celeste
                radius = 12
                width = 4

                # rettangolo arrotondato come contorno, leggermente dentro i bordi
                draw.rounded_rectangle(
                    [width // 2, width // 2, canvas.width - width // 2 - 1, canvas.height - width // 2 - 1],
                    radius=radius,
                    outline=outline_color,
                    width=width,
                )

            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{encoded}"

def team_selector(teams, logo_folder="logos"):
        teams = list(teams)

        if "selected_team" not in st.session_state:
            st.session_state.selected_team = teams[0]


        images = [
            image_to_uri(
                os.path.join(logo_folder, f"{team}.png"),
                selected=(team == st.session_state.selected_team),
            )
            for team in teams
        ]

        clicked = clickable_images(
            paths=images,
            titles=teams,
            div_style={
                "display": "grid",
                "grid-template-columns": "repeat(auto-fit, minmax(80px, 1fr))",
                "gap": "12px",
                "justify-items": "center",
                "align-items": "center",
                "width": "100%",
            },
            img_style={
                "width": "72px",
                "height": "72px",
                "padding": "8px",
                "border-radius": "12px",
                "cursor": "pointer",
                "transition": "transform .15s",
            },
            key="team_selector",
        )

        if clicked > -1 and teams[clicked] != st.session_state.selected_team:
            st.session_state.selected_team = teams[clicked]
            st.rerun()

        return st.session_state.selected_team

def cleanDataset(df, elo=False, minute=True):
  df = df.dropna(subset=['xg'])
  df = df.loc[df['position'] != 'G']
  # One-Hot Encoding
  x = pd.get_dummies(df, columns=['position'], prefix='position', drop_first=False)
  x = pd.get_dummies(x, columns=['situation'], prefix='situation', drop_first=False)
#   x = pd.get_dummies(x, columns=['bodyPart'], prefix='bodyPart', drop_first=False)
  x = pd.get_dummies(x, columns=['body_part'], prefix='bodyPart', drop_first=False)



  # Normalizations
  if minute == True:
    x['minute'] = x['minute']/90
  x['rating'] = (x['rating'].astype(int))/100
  if elo == True:
    x['eloTeam'] = x['eloTeam']/2336
  x['keeperRating'] = (x['keeperRating'].astype(int))/100
  if elo == True:
    x['eloOpponent'] = x['eloOpponent']/2336
  x['distance'] = x['distance']/105
  x['angle'] = x['angle']/90

  # Conversions
  x['position_D'] = x['position_D'].astype(int) # Defender
  x['position_F'] = x['position_F'].astype(int) # Forward
  x['position_M'] = x['position_M'].astype(int) # Midfielder

  x['situation_assisted'] = x['situation_assisted'].astype(int)
  x['situation_corner'] = x['situation_corner'].astype(int)
  x['situation_fast-break'] = x['situation_fast-break'].astype(int)
  x['situation_free-kick'] = x['situation_free-kick'].astype(int)
  x['situation_regular'] = x['situation_regular'].astype(int)
  x['situation_set-piece'] = x['situation_set-piece'].astype(int)
  x['situation_throw-in-set-piece'] = x['situation_throw-in-set-piece'].astype(int)
  # x['situation_Penalty'] = x['situation_Penalty'].astype(int)

  x['bodyPart_head'] = x['bodyPart_head'].astype(int)
  x['bodyPart_weakFoot'] = x['bodyPart_weakFoot'].astype(int)
  x['bodyPart_other'] = x['bodyPart_other'].astype(int)
  x['bodyPart_strongFoot'] = x['bodyPart_strongFoot'].astype(int)

  y = df['goal']
  x['isHome'] = x['isHome'].astype(int)
#   x = x.drop(columns=['goal', 'player', 'team', 'keeper', 'opponent', 'x', 'y', 'xg', 'homeTeam', 'awayTeam', 'index', 'round'])
  if 'week' in x.columns:
    x = x.drop(columns=['week'])
  if 'round' in x.columns:
    x = x.drop(columns=['round'])
  if 'Unnamed: 0' in x.columns:
    x = x.drop(columns=['Unnamed: 0'])
  if 'date' in df.columns and 'teamId' in df.columns:
    x = x.drop(columns=['game_index', 'date', 'home_team', 'away_team', 'team', 'teamId', 'player', 'playerId', 'keeper', 'opponent', 'x', 'y', 'xg', 'goal'])
  else:
    x = x.drop(columns=['game_index', 'home_team', 'away_team', 'team', 'player', 'player_id', 'keeper', 'opponent', 'x', 'y', 'xg', 'goal'])
  return x, y

def getXTrain(df, elo=False, minute=True, over='none', k=15, sampling_strategy='none', test_size=0.2):
    df = df.dropna(subset=['xg'])
    df = df[df.situation != "penalty"]
    x,y = cleanDataset(df, elo=elo, minute=minute)
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=42)
    if(over == "random"):
        oversample = RandomOverSampler(sampling_strategy=0.25, random_state=42)
        x,y = oversample.fit_resample(x,y)
    elif(over == "smote"):
        if(k!=0):
            smt = SMOTE(k_neighbors=k, random_state=42)
        else:
            if(sampling_strategy=='none'):
                smt = SMOTE(random_state=42)
            else:
                smt = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        x,y = smt.fit_resample(x,y)
    elif(over == "adasyn"):
        ada = ADASYN(random_state=42)
        x,y = ada.fit_resample(x,y)
    
    
    return X_train, x

def predictLocalGame(homeTeam, awayTeam, model, elo=False, minute=True, specific=False):
  if optionMenu1 == "Serie A":
    # allShots = pd.read_csv('datasets/seriea2425_id.csv')
    allShots = pd.read_csv('datasets/seriea_2526.csv')
  elif optionMenu1 == "Premier League":
    allShots = pd.read_csv('datasets/bpl2425_id.csv')
  elif optionMenu1 == "La Liga":
    allShots = pd.read_csv('datasets/liga2425_id.csv')
  elif optionMenu1 == "Bundesliga":
    allShots = pd.read_csv('datasets/bundes2425_id.csv')
  elif optionMenu1 == "Ligue 1":
    allShots = pd.read_csv('datasets/ligue12425_id.csv')
  if 'Unnamed: 0.1' in allShots.columns:
    allShots = allShots.drop(columns=['Unnamed: 0.1'])
  if 'Unnamed: 0' in allShots.columns:
    allShots = allShots.drop(columns=['Unnamed: 0'])
  if 'playerID' in allShots.columns:
    allShots = allShots.drop(columns=['playerID'])
  if 'keeperId' in allShots.columns:
    allShots = allShots.drop(columns=['keeperId'])
#   allShots = allShots.drop(columns=['playerID', 'keeperID'])
  shotmap = allShots.loc[(allShots['home_team'] == homeTeam) & (allShots['away_team'] == awayTeam)]
  shotmap = shotmap.reset_index()
  if 'level_0' in shotmap.columns:
      shotmap = shotmap.drop(columns=['level_0'])
  if minute==False:
    shotmap = shotmap.drop(columns=['minute'])
  shotmap = shotmap.drop(columns=['index', 'round', 'home_team', 'away_team'])
  shotmap = shotmap.dropna(subset=['xg'])
  shotmap = shotmap.loc[shotmap['position'] != 'G']
  if elo==False:
    shotmap = shotmap.drop(columns=['eloTeam', 'eloOpponent'])

  homeShots = shotmap.loc[shotmap['isHome'] == True]
#   print(np.unique(homeShots['situation']))
  homeShots = homeShots.reset_index()
  homeShots = homeShots.drop(columns=['index'])
  homeShots_p = homeShots.loc[homeShots["situation"] == "penalty"].copy()
  homeShots = homeShots.loc[homeShots['situation'] != 'penalty']
  awayShots = shotmap.loc[shotmap['isHome'] == False]
  awayShots = awayShots.reset_index()
  awayShots = awayShots.drop(columns=['index'])
  awayShots_p = awayShots.loc[awayShots["situation"] == "penalty"].copy()
  awayShots = awayShots.loc[awayShots['situation'] != 'penalty']
  
  if specific != True:
    df = pd.read_csv('datasets/top5_joined_id.csv')
    df = df.drop(columns=['playerID', 'keeperID'])
    if 'Unnamed: 0.1' in df.columns:
        df = df.drop(columns=['Unnamed: 0.1'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
  else:
    if optionMenu1 == "Serie A":
        # df = pd.read_csv('datasets/seriea_joined_new.csv')
        df = pd.read_csv('datasets/sofascore/seriea_2425.csv')
        df = df.drop(columns=['Unnamed: 0'])
    elif optionMenu1 == "Premier League":
        df = pd.read_csv('datasets/bpl_joined_id.csv')
        df = df.drop(columns=['Unnamed: 0', 'playerID', 'keeperID'])
    elif optionMenu1 == "La Liga":
        df = pd.read_csv('datasets/liga_joined_id.csv')
        df = df.drop(columns=['Unnamed: 0', 'playerID', 'keeperID'])
    elif optionMenu1 == "Bundesliga":
        df = pd.read_csv('datasets/bundes_joined_id.csv')
        df = df.drop(columns=['Unnamed: 0', 'playerID', 'keeperID'])
    elif optionMenu1 == "Ligue 1":
        df = pd.read_csv('datasets/ligue1_joined_id.csv')
        df = df.drop(columns=['Unnamed: 0', 'playerID', 'keeperID'])
  if minute==False:
    df = df.drop(columns=['minute'])
  if elo==False:
    df = df.drop(columns=['eloTeam', 'eloOpponent'])
  df_homeShots = pd.concat([df, homeShots]).reset_index()
  if 'level_0' in df_homeShots.columns:
      df_homeShots = df_homeShots.drop(columns=['level_0'])
  df_homeShots = df_homeShots.loc[df_homeShots['situation'] != 'penalty'].reset_index()
  if 'level_0' in df_homeShots.columns:
      df_homeShots = df_homeShots.drop(columns=['level_0'])
  df_x, df_y = cleanDataset(df_homeShots, elo=elo, minute=minute)

  homeShots_clean = df_x.tail(len(homeShots))
#   homeShots_clean = homeShots_clean.drop(columns=['index'])
  homeShots_clean = homeShots_clean.reset_index()
  homeShots_clean = homeShots_clean.drop(columns=['index'])
  if 'level_0' in homeShots_clean:
      homeShots_clean = homeShots_clean.drop(columns='level_0')
#   print(homeShots_clean['goal_difference'])
#   homeShots_clean
  homeXgPred = model.predict_proba(homeShots_clean)[:, 1]

  homePred = model.predict(homeShots_clean)
  homeShots['goalPred'] = homePred
  homeShots['xgPred'] = homeXgPred
  homeShots_p['xgPred'] = 0.79
  homeShots_p['goalPred'] = 1
  if len(homeShots_p)>0:
    homeShots = pd.concat([homeShots, homeShots_p])
  for i in homeShots.index:
    # if (homeShots.loc[i]['situation'] == 'penalty'):
    #   homeShots.at[i, 'xgPred'] = 0.75
    if (homeShots.loc[i]['xgPred'] == 0):
      homeShots.at[i, 'xgPred'] = 0.01
  homeShots['diff'] = homeShots['xgPred']-homeShots['xg']

  df_awayShots = pd.concat([df, awayShots]).reset_index()
  if 'level_0' in df_awayShots.columns:
      df_awayShots = df_awayShots.drop(columns=['level_0'])
  df_awayShots = df_awayShots.loc[df_awayShots['situation'] != 'penalty'].reset_index()
  if 'level_0' in df_awayShots.columns:
      df_awayShots = df_awayShots.drop(columns=['level_0'])
  df_x, df_y = cleanDataset(df_awayShots, elo=elo, minute=minute)
  awayShots_clean = df_x.tail(len(awayShots))
#   awayShots_clean = awayShots_clean.drop(columns=['index'])
  awayShots_clean = awayShots_clean.reset_index()
  awayShots_clean = awayShots_clean.drop(columns=['index'])
  if 'level_0' in awayShots_clean:
      awayShots_clean = awayShots_clean.drop(columns='level_0')
  awayXgPred = model.predict_proba(awayShots_clean)[:, 1]
  awayPred = model.predict(awayShots_clean)
  awayShots['goalPred'] = awayPred
  awayShots['xgPred'] = awayXgPred
  awayShots_p['xgPred'] = 0.79
  awayShots_p['goalPred'] = 1
  for i in awayShots.index:
    # if (awayShots.loc[i]['situation'] == 'penalty'):
    #   awayShots.at[i, 'xgPred'] = 0.75
    if (awayShots.loc[i]['xgPred'] == 0):
      awayShots.at[i, 'xgPred'] = 0.01
  awayShots['diff'] = awayShots['xgPred']-awayShots['xg']





  stats = {
      "shotmap": shotmap,
      "homeShots": homeShots,
      "awayShots": awayShots,
      "homeShots_clean": homeShots_clean,
      "awayShots_clean": awayShots_clean,
      "homeXgPred": homeXgPred,
      "homePred": homePred,
      "awayXgPred": awayXgPred,
      "awayPred": awayPred
  }
  return stats

def plotShots(teamShots):
    with st.expander("Visualize the Shotmap for the specific team in the match"):
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#22312b', half=True)
        fig,axs = pitch.draw(figsize=(8,4), ncols=2)
        fig.set_facecolor('#22312b')
        axs[0].patch.set_facecolor('#22312b')
        axs[0].set_title("Sofascore xG", color="white")
        axs[1].patch.set_facecolor('#22312b')
        axs[1].set_title("Model xG", color="white")

        legend1 = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='Strong Foot'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Weak Foot'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=6, label='Head'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='Other')
        ]
        
        legend2 = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='Model xG > Sofascore xG'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=6, label='Model xG = Sofascore xG'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Model xG < Sofascore xG')
        ]

        axs[0].legend(handles=legend1, loc='lower center', title='Legenda', fontsize='small', title_fontsize='small')
        axs[1].legend(handles=legend2, loc='lower center', title='Legenda', fontsize='small', title_fontsize='small')
        descriptions = []
        for i in teamShots.index:
            if(teamShots.loc[i]['goal'] == 0):
                shotOutcome = ''
            else:
                shotOutcome = ' (Goal)'
            description = str(i+1) + ' - ' + teamShots.loc[i]['player'] + ' - ' + ' (' + str(round(teamShots.loc[i]['xg'], 2)) + ' xG)' + shotOutcome
            # print(shotDescription)
            descriptions.append(description)
            x = 120-teamShots.loc[i]['x']
            y = 0.8 * teamShots.loc[i]['y']
            if(teamShots.loc[i]['body_part'] == 'strongFoot'):
                color='green'
            elif(teamShots.loc[i]['body_part'] == 'weakFoot'):
                color='red'
            elif(teamShots.loc[i]['body_part'] == 'head'):
                color='yellow'
            elif(teamShots.loc[i]['body_part'] == 'other'):
                color='blue'
            pitch.scatter(
                x=x, 
                y=y,
                ax=axs[0],
                s = shotsMultiplier*teamShots.loc[i]['xg'],
                c=color,
                edgecolors='white')
            if(teamShots.loc[i]['xg']<teamShots.loc[i]['xgPred']):
                color='green'
            elif(teamShots.loc[i]['xg']>teamShots.loc[i]['xgPred']):
                color='red'
            else:
                color='yellow'
            pitch.scatter(
                x=x, 
                y=y,
                ax=axs[1],
                s = shotsMultiplier*teamShots.loc[i]['xgPred'],
                c=color,
                edgecolors='white')
        st.pyplot(fig)

    # shotIndex = drawPitch(teamShots)
    return descriptions

def drawPitch(teamShots, length=120, width=80):
    fig = go.Figure()

    halfway = length / 2  # 60

    # tutte le coordinate qui sono già in "spazio plot": x_plot = pitch_y, y_plot = pitch_x
    lines = [
        # bordo del mezzo campo
        dict(x0=0, y0=halfway, x1=0, y1=length),              # lato sinistro
        dict(x0=width, y0=halfway, x1=width, y1=length),      # lato destro
        dict(x0=0, y0=length, x1=width, y1=length),           # linea di fondo
        dict(x0=0, y0=halfway, x1=width, y1=halfway),         # linea di metà campo

        # area di rigore (18 x 44, statsbomb: da y=18 a y=62 -> centrata su width/2)
        dict(x0=18, y0=length, x1=18, y1=length - 18),
        dict(x0=62, y0=length, x1=62, y1=length - 18),
        dict(x0=18, y0=length - 18, x1=62, y1=length - 18),

        # area piccola (6 x 20, da y=30 a y=50)
        dict(x0=30, y0=length, x1=30, y1=length - 6),
        dict(x0=50, y0=length, x1=50, y1=length - 6),
        dict(x0=30, y0=length - 6, x1=50, y1=length - 6),
    ]

    for l in lines:
        fig.add_shape(type="line", line=dict(color="white", width=2), **l)

    # dischetto del rigore (12 yard dalla linea di fondo)
    fig.add_shape(
        type="circle",
        x0=width/2 - 0.5, y0=length - 12 - 0.5,
        x1=width/2 + 0.5, y1=length - 12 + 0.5,
        fillcolor="white", line=dict(color="white"),
    )

    # arco del dischetto (facoltativo, semplificato come cerchio pieno sopra è già ok per il puntino;
    # se vuoi anche l'arco della "D" fuori area, dimmelo e te lo aggiungo)

    def add_penalty_arc(fig, spot_x=12, box_edge_x=18, spot_y=40, radius=9.15, n_points=50):
        """
        spot_x: distanza del dischetto dalla porta (12 yard)
        box_edge_x: distanza del bordo area dalla porta (18 yard)
        spot_y: centro larghezza campo (40)
        """
        half_width = np.sqrt(radius**2 - (box_edge_x - spot_x)**2)

        y_range = np.linspace(spot_y - half_width, spot_y + half_width, n_points)
        x_range = spot_x + np.sqrt(radius**2 - (y_range - spot_y)**2)  # + invece di -

        arc_x_plotly = y_range
        arc_y_plotly = 120 - x_range

        fig.add_trace(go.Scatter(
            x=arc_x_plotly,
            y=arc_y_plotly,
            mode="lines",
            line=dict(color="white", width=2),
            hoverinfo="skip",
            showlegend=False,
        ))
    add_penalty_arc(fig)

    x_range = [-2, width + 2]
    y_range = [halfway - 2, length + 2]

    x_span = x_range[1] - x_range[0]   # 84
    y_span = y_range[1] - y_range[0]   # 64

    plot_height = 600
    plot_width = plot_height * (x_span / y_span)

    # fig.update_layout(
    #     plot_bgcolor="#22312b",
    #     xaxis=dict(range=[-2, width + 2], visible=False),
    #     yaxis=dict(range=[halfway - 2, length + 2], visible=False, scaleanchor="x"),
    #     margin=dict(l=0, r=0, t=0, b=0),
    #     height=600,
    # )

    fig.update_layout(
        plot_bgcolor="#0E1117",
        xaxis=dict(range=x_range, visible=False, constrain="domain"),
        yaxis=dict(range=y_range, visible=False, scaleanchor="x", constrain="domain"),
        margin=dict(l=0, r=0, t=0, b=0),
        width=plot_width,
        height=plot_height,
    )

    # --- coordinate dei tiri ---
    x_plotly = 0.8 * teamShots["y"]   # larghezza statsbomb (0-80) -> asse orizzontale
    y_plotly = 119.5 - teamShots["x"]   # lunghezza statsbomb (0-120) -> asse verticale, porta in alto
    
    # st.write(teamShots)

    min_size = 8
    max_size = 30

    xg = teamShots["xg"]
    marker_sizes = min_size + (xg - xg.min()) / (xg.max() - xg.min()) * (max_size - min_size)

    fig.add_trace(go.Scatter(
        x=x_plotly,
        y=y_plotly,
        mode="markers",
        marker=dict(size=marker_sizes, color="red", line=dict(color="white", width=1)),
        customdata=teamShots.index,
        hovertext=teamShots["description"],
        hoverinfo="text",
    ))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        event = st.plotly_chart(
            fig,
            on_select="rerun",
            selection_mode="points",
            key="pitch_chart",
            use_container_width=False,
        )

    # print(event['selection']['points'][0]['customdata'])
    # print(event['selection'])
    if event['selection']['points'] != []:
        # print(event['selection']['points'])
        shotIndex = event['selection']['points'][0]['customdata']
        shot = teamShots.loc[shotIndex]
        # print(shot['situation'])
        if shot['situation'] != 'penalty':
            penalty = False
        else:
            penalty = True
        # st.write("return")
        return shotIndex, penalty
    return None, None


def plotShap(shapValues, elo, shot, predicted_xg):
    # print("shot:", shot)
    features = []
    shap_values = []
    values = []
    dict = []
    # shapValues = shapValues[1:]
    # shot = shot[1:]
    # print(shapValues)
    for (i, feature) in enumerate(shapValues.data.index):
        # print(i,feature)
        features.append(feature)
        shapValue = round(shapValues.values[i], 2)
        shap_values.append(shapValue)
        # print(feature, shapValue)
        dict.append({
            "Feature": feature,
            "ShapValue": shapValue
        })
    if elo == True:
        for (i, value) in enumerate(shot):
            # print(i, value)
            if(i==0):
                value = int(value*90)   #minuto
            elif (i==8):
                value = int(value)  #differenza goal
            elif(i==3 or i==4):
                value = int(value*100)  #ratings
            elif(i==2 or i==5):
                value = int(value*2168) #elos
            elif(i==6):
                value = value*105   #distanza
            elif(i==7):
                value = value*90    #angolo
            else:
                if i!=0:
                    value = int(value)
            values.append(value)
            dict[i]["Value"] = value
    else:
       for (i, value) in enumerate(shapValues.data):
            # print(i, value)
            if(i==0):
                value = int(value*90)   #minuto
            elif (i==1):
                value = int(value)  #differenza goal
            elif(i==2 or i==3):
                value = int(value*100)  #ratings
            elif(i==5):
                value = int(value*120)   #distanza
            elif(i==6):
                value = int(value*90)    #angolo
            else:
                value = int(value)
            values.append(value)
    # print(features)
    # print(values)
    # print(dict)
    if(elo == True):
        features = ['Minute','Plays Home', 'Team Elo', 'Shooter Rating', 'Keeper Rating', 'Opponent Elo', 'Distance', 'Angle', 'Goal Difference', 'Position - Defender', 'Position - Forward', 'Position - Midfielder', 'Situation - Assisted', 'Situation - Corner Kick', 'Situation - Fast-Break', 'Situation - Free Kick', 'Situation - Regular', 'Situation - Set Piece', 'Situation - Throw-In', 'Body Part - Head', 'Body Part - Other', 'Body Part - Strong Foot', 'Body Part - Weak Foot']
    else:
        features = ['Minute','Goal Difference', 'Shooter Rating', 'Keeper Rating', 'Plays Home', 'Distance', 'Angle', 'Position - Defender', 'Position - Forward', 'Position - Midfielder', 'Situazione - Assisted', 'Situation - Corner Kick', 'Situation - Fast-Break', 'Situation - Free Kick', 'Situation - Regular', 'Situation - Set Piece', 'Situation - Throw-In', 'Body Part - Head', 'Body Part - Other', 'Body Part - Strong Foot', 'Body Part - Weak Foot']
    features_values = []
    if(elo==True):
        for i in range(0, len(features)):
            if(i<=9 and i!=1):
                desc = str(features[i]) + ': ' + str(round(values[i], 2))
                # features_values.append(str(features[i]) + ': ' + str(round(values[i], 2)))
            else:
                if(values[i] == 1):
                    desc = str(features[i]) + ': Yes'
                    # features_values.append(str(features[i]) + ': Yes')
                else:
                    desc = str(features[i]) + ': No'
                    # features_values.append(str(features[i]) + ': No')
            features_values.append(desc)
            dict[i]['Description'] = desc
    else:
        for i in range(0, len(features)):
            if(i<=6 and i!=4):
                features_values.append(str(features[i]) + ': ' + str(round(values[i], 2)))
            else:
                if(values[i] == 1):
                    features_values.append(str(features[i]) + ': Yes')
                else:
                    features_values.append(str(features[i]) + ': No')
    # print(features_values)
    # print(dict)
    dictDF = pd.DataFrame(dict)
    # print(dictDF)
    # st.write(pd.DataFrame(dict))
    

    top10 = dictDF.reindex(dictDF["ShapValue"].abs().sort_values(ascending=False).index).head(10)
    shapvalues_array = top10["ShapValue"].to_numpy()[::-1]
    descriptions_array = top10["Description"].to_numpy()[::-1]
    

    # Base value SHAP (log-odds)
    base_value = float(shapValues.base_values[0])
    # SHAP values delle feature
    shap = shapValues.values
    # Nomi descrizioni feature
    descriptions = dictDF["Description"].to_numpy()
    # xG iniziale (valore medio del modello)
    base_xg = expit(base_value)
    # xG finale previsto
    final_xg = expit(base_value + np.sum(shap))
    # Ordino le feature per importanza assoluta SHAP
    order = np.argsort(np.abs(shap))[::-1]
    # Costruzione waterfall in xG
    current_logit = base_value
    current_xg = base_xg

    waterfall = []
    for idx in order:
        next_logit = current_logit + shap[idx]
        next_xg = expit(next_logit)

        waterfall.append({
            "Description": descriptions[idx],
            "Impact": next_xg - current_xg,
            "Feature": idx
        })
        current_logit = next_logit
        current_xg = next_xg

    waterfall = pd.DataFrame(waterfall)
    # tengo le 10 feature più importanti
    top10 = waterfall.head(10)
    
    with st.expander("How each feature changed the xG"):

        st.info(
            """
            **How does this explanation work?**

            The model starts from the average xG value of all shots in the dataset.
            This represents the expected probability of scoring for a typical shot.

            Each feature of this shot (for example distance, angle, player quality or game situation)
            then increases or decreases this value according to its contribution.

            The final xG is the result of combining all these effects.
            Green bars show features that increase the probability of scoring,
            while red bars show features that reduce it.
            """
        )

        st.write(
            f"Average xG: **{base_xg:.2f}**  →  "
            # f"Predicted xG: **{final_xg:.2f}**"
            f"Predicted xG: **{predicted_xg:.2f}**"
        )


        fig = plt.figure(figsize=(9,6))


        impacts = top10["Impact"].to_numpy()[::-1]
        labels = top10["Description"].to_numpy()[::-1]


        bars = plt.barh(
            labels,
            impacts,
            left=base_xg,
            color=[
                "green" if x > 0 else "red"
                for x in impacts
            ]
        )


        plt.axvline(
            base_xg,
            color="white",
            linestyle="--",
            linewidth=0.8
        )

        plt.text(
            base_xg,
            plt.ylim()[1],
            f"Avg xG\n{base_xg:.2f}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=10
        )

        plt.axvline(
            final_xg,
            color="yellow",
            linestyle=":",
            linewidth=1
        )

        plt.text(
            final_xg,
            plt.ylim()[1],
            # f"Final xG\n{final_xg:.2f}",
            f"Final xG\n{predicted_xg:.2f}",
            color="yellow",
            ha="center",
            va="bottom",
            fontsize=10
        )


        plt.xlabel(
            "Change in xG",
            color="white"
        )

        plt.title(
            "Contribution of each feature",
            color="white",
            pad=35
        )


        plt.tick_params(
            axis='both',
            colors='white'
        )
        ax = plt.gca()

        ax.tick_params(axis='y', pad=25, length=0)



        ax = plt.gca()

        for pos in [
            'right',
            'top',
            'bottom',
            'left'
        ]:
            ax.spines[pos].set_visible(False)


        # valori sulle barre
        for bar, value in zip(bars, impacts):

            plt.text(
                base_xg + value + (0.001 if value > 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f"{value:+.3f}",
                va="center",
                ha="left" if value > 0 else "right",
                color="white",
                fontsize=10
            )


        st.pyplot(
            fig,
            transparent=True
        )

    

def showShots():
    # useSpecific = st.checkbox("Use a League-Specific Model")
    useSpecific = True
    if useSpecific != True:
        df = pd.read_csv('datasets/top5_joined_id.csv')
        df = df.drop(columns=['Unnamed: 0', 'playerID', 'keeperID'])
    else:
        if optionMenu1 == "Serie A":
            # df = pd.read_csv('datasets/seriea_joined_new.csv')
            df = pd.read_csv('datasets/sofascore/seriea_2425.csv')
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            # st.dataframe(df)
        elif optionMenu1 == "Premier League":
            df = pd.read_csv('datasets/bpl_joined_id.csv')
            # print(df.columns)
            df = df.drop(columns=['Unnamed: 0', 'playerID', 'keeperID'])
            # st.dataframe(df)
        elif optionMenu1 == "La Liga":
            df = pd.read_csv('datasets/liga_joined_id.csv')
            # print(df.columns)
            df = df.drop(columns=['playerID', 'keeperID'])
        elif optionMenu1 == "Bundesliga":
            df = pd.read_csv('datasets/bundes_joined_id.csv')
            # print(df.columns)
            df = df.drop(columns=['playerID', 'keeperID'])
        elif optionMenu1 == "Ligue 1":
            df = pd.read_csv('datasets/ligue1_joined_id.csv')
            # print(df.columns)
            df = df.drop(columns=['playerID', 'keeperID'])
    # useElo = st.checkbox("Use the teams' Elo Ratings")
    useElo = True
    if useElo == True:
        elo = True
        global modelName
        if optionMenu1 == "Serie A":
            # modelName = 'ITA_full'
            modelName = 'ITA_2425'
        elif optionMenu1 == "Premier League":
            modelName = 'ENG_full'
        elif optionMenu1 == "La Liga":
            modelName = 'ESP_full'
        elif optionMenu1 == "Bundesliga":
            modelName = 'GER_full'
        elif optionMenu1 == "Ligue 1":
            modelName = 'FRA_full'
    else:
        elo = False
        if optionMenu1 == "Serie A":
            modelName = 'ITA_minute'
        elif optionMenu1 == "Premier League":
            modelName = 'ENG_minute'
        elif optionMenu1 == "La Liga":
            modelName = 'ESP_minute'
        elif optionMenu1 == "Bundesliga":
            modelName = 'GER_minute'
        elif optionMenu1 == "Ligue 1":
            modelName = 'FRA_minute'
        df = df.drop(columns=['eloTeam', 'eloOpponent'])
    if useSpecific != True:
        model = joblib.load('models/TOP5_' + modelName + '.sav')
    else:
        model = joblib.load('models/' + modelName + '.sav')

    X_train, y = getXTrain(df, elo=elo, minute=True)
    explainer = shap.Explainer(model, X_train)




    if useSpecific != True:
        shotsDF = pd.read_excel('allShots/allShots_TOP5_' + modelName + '.xlsx')
    else:
        shotsDF = pd.read_excel('allShots/allShots_' + modelName + '.xlsx')
    shotsDF = shotsDF.drop(columns='Unnamed: 0')
    if useSpecific != True:
        statsDF = pd.read_excel('leagueStats/leagueStats_TOP5_' + modelName + '.xlsx')
    else:
        statsDF = pd.read_excel('leagueStats/leagueStats_' + modelName + '.xlsx')
    statsDF = statsDF.drop(columns='Unnamed: 0')


    if optionMenu1 == "Serie A":
        # schedule = pd.read_csv('serieaSchedule.csv')
        schedule = pd.read_csv('schedules/seriea.csv')
    elif optionMenu1 == "Premier League":
        schedule = pd.read_csv('bplSchedule.csv')
    elif optionMenu1 == "La Liga":
        schedule = pd.read_csv('ligaSchedule.csv')
    elif optionMenu1 == "Bundesliga":
        schedule = pd.read_csv('bundesSchedule.csv')
    elif optionMenu1 == "Ligue 1":
        schedule = pd.read_csv('ligue1Schedule.csv')

    
    teams = np.unique(schedule['home_team'])
    # scheduleTeam = st.selectbox("Select a Team", teams, index=None)
    
    scheduleTeam = team_selector(teams)

    if scheduleTeam:
        if 'Unnamed: 0' in schedule.columns:
            schedule = schedule.drop(columns='Unnamed: 0')
        scheduleDone = schedule[schedule['home_score'].notna()]
        # scheduleDone = schedule.loc[schedule['week']<=lastRound]
        scheduleDone = scheduleDone.loc[(scheduleDone['home_team'] == scheduleTeam) | (scheduleDone['away_team'] == scheduleTeam)]
        descriptions = []
        for i in scheduleDone.index:
            description = 'Round ' + str(scheduleDone.loc[i]['round']) + ': ' +  scheduleDone.loc[i]['home_team'] + ' - ' + scheduleDone.loc[i]['away_team'] + ' ' + str(int(scheduleDone.loc[i]['home_score'])) + ' - ' + str(int(scheduleDone.loc[i]['away_score']))
            descriptions.append(description)
        scheduleDone['description'] = descriptions
        gameDescription = st.selectbox('Select a Match', scheduleDone['description'], index=None)
        if gameDescription:
            gameIndex = scheduleDone.loc[scheduleDone['description'] == gameDescription].index[0]
            homeTeam = scheduleDone.loc[gameIndex]['home_team']
            awayTeam = scheduleDone.loc[gameIndex]['away_team']
            homeXg = round(statsDF.loc[gameIndex]['homeXg'], 2)
            awayXg = round(statsDF.loc[gameIndex]['awayXg'], 2)
            homeXgPred = round(statsDF.loc[gameIndex]['homeXgPred'], 2)
            awayXgPred = round(statsDF.loc[gameIndex]['awayXgPred'], 2)
            st.write("Sofascore xG:")
            displayScore(homeXg, awayXg, homeTeam, awayTeam)
            st.write("Model xG:")
            displayScore(homeXgPred, awayXgPred, homeTeam, awayTeam)
            # st.error("Sofascore xG: " + str(statsDF.loc[gameIndex]['homeXg']) + ' - ' + str(statsDF.loc[gameIndex]['awayXg']))
            # st.info("Model xG: " + str(statsDF.loc[gameIndex]['homeXgPred']) + ' - ' + str(statsDF.loc[gameIndex]['awayXgPred']))
            stats = predictLocalGame(homeTeam, awayTeam, model, elo=elo, minute=True, specific=useSpecific)
            # gameShots = shotsDF.loc[shotsDF['gameIndex'] == gameIndex]
            
            gameShots = shotsDF.loc[(shotsDF['home_team'] == homeTeam) & (shotsDF['away_team'] == awayTeam)]
            
            
            home_team = scheduleDone.loc[gameIndex]['home_team']
            away_team = scheduleDone.loc[gameIndex]['away_team']

            selectedTeam = st.selectbox('Select a Team', [home_team, away_team], index=None)

            if selectedTeam:
                teamShots = gameShots.loc[gameShots['team'] == selectedTeam].reset_index(drop=True)
            
                teamShots['description'] = plotShots(teamShots)
                shotIndex, penalty = drawPitch(teamShots)
                # shotDescription = st.selectbox('Select a Shot', teamShots['description'], index=None)
                # if shotDescription:
                if shotIndex is None:
                    st.info("Select a Shot from the shotmap to see its details")
                else:
                    # shotIndex = teamShots.loc[teamShots['description'] == shotDescription].index[0]
                    
                    # st.error("Sofascore xG: " + str(teamShots.loc[shotIndex]['xg']))
                    # st.info("Model xG: " + str(teamShots.loc[shotIndex]['xgPred']))
                    displayXg(round(teamShots.loc[shotIndex]['xg'], 2), round(teamShots.loc[shotIndex]['xgPred'], 2))
                    
                    # print(len(stats['homeShots_clean']), len(stats['awayShots_clean']))
                    # print(stats['homeShots_clean'], stats['awayShots_clean'])
                    if(selectedTeam == home_team):
                        shot = stats['homeShots_clean'].loc[shotIndex]
                    elif(selectedTeam == away_team):
                        # print(stats['awayShots_clean'])
                        shot = stats['awayShots_clean'].loc[shotIndex]
                    # print(shot)
                    if not penalty:
                        shapValues = explainer(shot, check_additivity=False)
                        plotShap(shapValues, elo, shot, round(teamShots.loc[shotIndex]['xgPred'], 2))
                    else:
                        st.info("**Penalties**: " \
                        "xG values for penalties are fixed, and most providers assign values between 0.75 and 0.80. For this project, a value of **0.79** has been used, the same of SofaScore.")
                    showViolinPlot(specific=useSpecific, elo=elo)


def showPlayers():
    # useSpecific = st.checkbox("Use a League-Specific Model")
    # useElo = st.checkbox("Use the teams' Elo Ratings")
    useSpecific = True
    useElo = True
    if useElo == True:
        elo = True
        if optionMenu1 == "Serie A":
            # modelName = 'ITA_full'
            modelName = 'ITA_2425'
        elif optionMenu1 == "Premier League":
            modelName = 'ENG_full'
        elif optionMenu1 == "La Liga":
            modelName = 'ESP_full'
        elif optionMenu1 == "Bundesliga":
            modelName = 'GER_full'
        elif optionMenu1 == "Ligue 1":
            modelName = 'FRA_full'
    else:
        elo = False
        if optionMenu1 == "Serie A":
            modelName = 'ITA_minute'
        elif optionMenu1 == "Premier League":
            modelName = 'ENG_minute'
        elif optionMenu1 == "La Liga":
            modelName = 'ESP_minute'
        elif optionMenu1 == "Bundesliga":
            modelName = 'GER_minute'
        elif optionMenu1 == "Ligue 1":
            modelName = 'FRA_minute'
    if useSpecific != True:
        shotsDF = pd.read_excel('allShots/allShots_TOP5_' + modelName + '.xlsx')
    else:
        shotsDF = pd.read_excel('allShots/allShots_' + modelName + '.xlsx')
    shotsDF = shotsDF.drop(columns=['Unnamed: 0'])
    
    photoStrikers(shotsDF)
    photoKeepers(shotsDF)

    

def photoStrikers(shotsDF):
    shotPlayers = np.unique(shotsDF['player'])
    players = []
    playerIDs = []
    xgSums = []
    xgPredSums = []
    goalSums = []
    for player in shotPlayers:
        playerShots = shotsDF.loc[shotsDF['player'] == player].reset_index()
        # print(playerShots)
        playerID = playerShots.loc[0]['playerID']
        xgSum = np.sum(playerShots['xg'])
        xgPredSum = np.sum(playerShots['xgPred'])
        goalSum = np.sum(playerShots['goal'])
        players.append(player)
        playerIDs.append(playerID)
        xgSums.append(xgSum)
        xgPredSums.append(xgPredSum)
        goalSums.append(goalSum)
    playersDF = pd.DataFrame()
    playersDF['Player'] = players
    playersDF['Player ID'] = playerIDs
    playersDF['Sofascore xG'] = xgSums
    playersDF['Model xG'] = xgPredSums
    playersDF['Goal'] = goalSums
    playersDF['Difference (Sofascore)'] = playersDF['Goal'] - playersDF['Sofascore xG']
    playersDF['Difference (Model)'] = playersDF['Goal'] - playersDF['Model xG']

    st.header("Movement Players")
    col1, col2 = st.columns(2)
    sofaOPdf = playersDF.sort_values(by="Difference (Sofascore)", ascending=False).drop(columns=['Model xG', 'Difference (Model)']).head(3)
    sofaOPdf = sofaOPdf.reset_index()
    modelOPdf = playersDF.sort_values(by="Difference (Model)", ascending=False).drop(columns=['Sofascore xG', 'Difference (Sofascore)']).head(3)
    modelOPdf = modelOPdf.reset_index()
    with col1:
        st.subheader("Overperformers based on Sofascore")
        # st.dataframe(sofaOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[0]['Player ID']) + "/image"
            p = sofaOPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(sofaOPdf.loc[0]['Goal'])
            pDiff = str(round(sofaOPdf.loc[0]['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'seagreen')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[1]['Player ID']) + "/image"
            p = sofaOPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(sofaOPdf.loc[1]['Goal'])
            pDiff = str(round(sofaOPdf.loc[1]['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'seagreen')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[2]['Player ID']) + "/image"
            p = sofaOPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(sofaOPdf.loc[2]['Goal'])
            pDiff = str(round(sofaOPdf.loc[2]['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'seagreen')
    with col2:
        st.subheader("Overperformers based on the Model")
        # st.dataframe(modelOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[0]['Player ID']) + "/image"
            p = modelOPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Model)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'forestgreen')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[1]['Player ID']) + "/image"
            p = modelOPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Model)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'forestgreen')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[2]['Player ID']) + "/image"
            p = modelOPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Model)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'forestgreen')
    
    col1, col2 = st.columns(2)
    sofaUPdf = playersDF.sort_values(by="Difference (Sofascore)", ascending=True).drop(columns=['Model xG', 'Difference (Model)']).head(3)
    sofaUPdf = sofaUPdf.reset_index()
    modelUPdf = playersDF.sort_values(by="Difference (Model)", ascending=True).drop(columns=['Sofascore xG', 'Difference (Sofascore)']).head(3)
    modelUPdf = modelUPdf.reset_index()
    with col1:
        st.subheader("Underperformers based on Sofascore")
        # st.dataframe(sofaUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[0]['Player ID']) + "/image"
            p = sofaUPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'firebrick')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[1]['Player ID']) + "/image"
            p = sofaUPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'firebrick')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[2]['Player ID']) + "/image"
            p = sofaUPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'firebrick')
    with col2:
        st.subheader("Underperformers based on the Model")
        # st.dataframe(modelUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        
        
        
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[0]['Player ID']) + "/image"
            p = modelUPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Model)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'red')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[1]['Player ID']) + "/image"
            p = modelUPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Model)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'red')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[2]['Player ID']) + "/image"
            p = modelUPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Model)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'red')


def photoKeepers(shotsDF):
    shotKeepers = np.unique(shotsDF['keeper'])
    keepers = []
    keeperIDs = []
    xgSums = []
    xgPredSums = []
    goalSums = []
    for keeper in shotKeepers:
        keeperShots = shotsDF.loc[shotsDF['keeper'] == keeper].reset_index()
        keeperID = keeperShots.loc[0]['keeperID']
        xgSum = np.sum(keeperShots['xg'])
        xgPredSum = np.sum(keeperShots['xgPred'])
        goalSum = np.sum(keeperShots['goal'])
        keepers.append(keeper)
        keeperIDs.append(keeperID)
        xgSums.append(xgSum)
        xgPredSums.append(xgPredSum)
        goalSums.append(goalSum)
    keepersDF = pd.DataFrame()
    keepersDF['Player'] = keepers
    keepersDF['Player ID'] = keeperIDs
    keepersDF['Sofascore xG'] = xgSums
    keepersDF['Model xG'] = xgPredSums
    keepersDF['Goal'] = goalSums
    keepersDF['Difference (Sofascore)'] = keepersDF['Sofascore xG'] - keepersDF['Goal']
    keepersDF['Difference (Model)'] = keepersDF['Model xG'] - keepersDF['Goal']
    
    st.header("Goalkeepers")

    col1, col2 = st.columns(2)
    sofaOPdf = keepersDF.sort_values(by="Difference (Sofascore)", ascending=False).drop(columns=['Model xG', 'Difference (Model)']).head(3)
    sofaOPdf = sofaOPdf.reset_index()
    modelOPdf = keepersDF.sort_values(by="Difference (Model)", ascending=False).drop(columns=['Sofascore xG', 'Difference (Sofascore)']).head(3)
    modelOPdf = modelOPdf.reset_index()
    with col1:
        st.subheader("Overperformers based on Sofascore")
        
        # st.dataframe(sofaOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[0]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaOPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'seagreen')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[1]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaOPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'seagreen')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaOPdf.loc[2]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaOPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(round(p['Difference (Sofascore)'],2))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'seagreen')
    with col2:
        st.subheader("Overperformers based on the Model")
        # st.dataframe(modelOPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[0]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelOPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Model)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'forestgreen')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[1]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelOPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Model)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'forestgreen')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelOPdf.loc[2]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelOPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Model)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (-" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'forestgreen')

    col1, col2 = st.columns(2)
    sofaUPdf = keepersDF.sort_values(by="Difference (Sofascore)", ascending=True).drop(columns=['Model xG', 'Difference (Model)']).head(3)
    sofaUPdf = sofaUPdf.reset_index()
    modelUPdf = keepersDF.sort_values(by="Difference (Model)", ascending=True).drop(columns=['Sofascore xG', 'Difference (Sofascore)']).head(3)
    modelUPdf = modelUPdf.reset_index()
    with col1:
        st.subheader("Underperformers based on Sofascore")
        # st.dataframe(sofaUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[0]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaUPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Sofascore)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'firebrick')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[1]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaUPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Sofascore)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'firebrick')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(sofaUPdf.loc[2]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = sofaUPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Sofascore xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Sofascore)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'firebrick')
    with col2:
        st.subheader("Underperformers based on the Model")
        # st.dataframe(modelUPdf, hide_index=True, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[0]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelUPdf.loc[0]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Model)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'red')
        with c2:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[1]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelUPdf.loc[1]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Model)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'red')
        with c3:
            playerUrl = "https://img.sofascore.com/api/v1/player/" + str(modelUPdf.loc[2]['Player ID']) + "/image"
            # response = requests.get(playerUrl)
            # image = Image.open(BytesIO(response.content))
            # output = remove(image)
            p = modelUPdf.loc[2]
            pName = str(p['Player'])
            name, surname = pName.split(' ', 1)
            pXG = str(round(p['Model xG'], 2))
            pGoal = str(p['Goal'])
            pDiff = str(abs(round(p['Difference (Model)'],2)))
            caption = pName + ", " + pXG + " xG" + ", " + pGoal + " Goal" + " (+" + pDiff + ")"
            # st.image(playerUrl, caption=caption)
            displayCard(playerUrl, name, surname, pXG, pGoal, pDiff, 'red')

def showViolinPlot(specific, elo):
    # print("Model + " + str(modelName))
    with st.expander("See how does the model reasons"):
        st.markdown("<h1 style='text-align: center;'>How Does the Model Think?</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if specific == True:
                st.image("violinPlots/" + modelName + ".png")
            else:
                if elo==True:
                    st.image("violinPlots/TOP5_full.png")
                else:
                    st.image("violinPlots/TOP5_minute.png")
        with col2:
            st.write("## Key Aspects:")
            match modelName:
                case "ITA_2425":
                    st.markdown("""
                    - **Fast-Breaks** are more effective and important than in other leagues, and they influence positively a shot's probability
                        - **Serie A is the league in which Fast-Breaks influence positively the most**
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                        - **Serie A is the league in which Corner Kicks influence negatively the most**
                    - Being a good player does not affect the shot's probability, but having a low shooting ability affects deeply the scoring chances
                        - At the same time, playing against a low-tier team affects positively the scoring chances
                    """)
                case "ITA_minute":
                    st.markdown("""
                    - **Fast-Breaks** are more effective and important than in other leagues, and they influence positively a shot's probability
                        - **Serie A is the league in which Fast-Breaks influence positively the most**
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                        - **Serie A is the league in which Corner Kicks influence negatively the most**
                    - Serie A is the only league in which the shot's **angle** does not have a clear trend
                    - A low value for the **minute** could affect negatively the probability, much more than in other leagues 
                        - That could implicate that teams "study themselves" more in the first minutes
                    - Serie A is the only league in which being in a **disadvantage** influences negatively a shot probability
                        - Having an advantage influences slightly positively the probabilities
                    """)
                case "ITA_full":
                    st.markdown("""
                    - **Fast-Breaks** are more effective and important than in other leagues, and they influence positively a shot's probability
                        - **Serie A is the league in which Fast-Breaks influence positively the most**
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                        - **Serie A is the league in which Corner Kicks influence negatively the most**
                    - A low value for the **minute** could affect negatively the probability, much more than in other leagues 
                        - That could implicate that teams "study themselves" more in the first minutes
                    - Serie A is the only league in which being in a **disadvantage** influences negatively a shot probability
                        - Having an advantage influences slightly positively the probabilities
                    - A high **ELO Rating** gives a slight advantage
                    """)
                case "ENG_minute":
                    st.markdown("""
                    - **Fast-Breaks** influence positively a shot's probability, but not as in other leagues
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                    - A low value for the **minute** does **not** affect the probabilities like in other leagues
                        - Teams do not "waste time" studying the opponent
                        - **Probabilities get higher in the last minutes**
                    - Having an **advantage** influences positively the probabilities
                        - Premier League is one of the few leagues in which **having a disadvantage does not influence negatively the probabilities**
                    - **Premier League is the league which influences most negatively set pieces**
                        - Premier League is also **the only league which does not influence positively free kicks**
                    """)

                case "ENG_full":
                    st.markdown("""
                    - **Fast-Breaks** influence positively a shot's probability, but not as in other leagues
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                    - A low value for the **minute** does **not** affect the probabilities like in other leagues
                        - Teams do not "waste time" studying the opponent
                        - **Probabilities get higher in the last minutes**
                    - Having an **advantage** influences positively the probabilities
                        - Premier League is one of the few leagues in which **having a disadvantage does not influence negatively the probabilities**
                    - **Premier League is the league which influences most negatively set pieces**
                        - Premier League is also **the only league which does not influence positively free kicks**
                    - **ELO Rating** for the **opposite** team does not provide a clear trend
                    """)

                case "ESP_minute":
                    st.markdown("""
                    - **Fast-Breaks** influence positively a shot's probability, but not as in other leagues
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                    - **Rating** does not influence as positively as in other leagues
                        - **Keeper Rating** is more important than in other leagues
                    - The **minute** feature does not show a clear trend
                    """)

                case "ESP_full":
                    st.markdown("""
                    - **Fast-Breaks** influence positively a shot's probability, but not as in other leagues
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                    - **Rating** does not influence as positively as in other leagues
                        - **Keeper Rating** is more important than in other leagues
                    - The **minute** feature does not show a clear trend
                    - The **ELO Ratings** for team and opponent influence more positively than in other leagues
                    """)

                case "GER_minute":
                    st.markdown("""
                    - **Fast-Breaks** influence positively a shot's probability, but not as in other leagues
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                        - **Bundesliga is the only league in which a head shot could also influence positively**
                    - The **minute** feature does not show a clear trend
                    - Having an **advantage** influences positively the probabilities
                        - Premier League is one of the few leagues in which **having a disadvantage does not influence negatively the probabilities**
                    - **Bundesliga is the league which influences most positively free kicks**
                    """)

                case "GER_full":
                    st.markdown("""
                    - **Fast-Breaks** influence positively a shot's probability, but not as in other leagues
                    - **Head Shots** and **Corner Kicks** influence negatively a shot's probability
                        - **Bundesliga is the only league in which a head shot could also influence positively**
                    - The **minute** feature does not show a clear trend
                    - Having an **advantage** influences positively the probabilities
                        - Premier League is one of the few leagues in which **having a disadvantage does not influence negatively the probabilities**
                    - **Bundesliga is the league which influences most positively free kicks**
                    """)
            
        

def displayCard(url, name, surname, xg, goal, diff, bgcolor):
    card_html = f"""
    <div class="card" style="background-color: {bgcolor}">
        <img src="{url}" alt="Immagine della card">
        <div class="card-title">{name}<br>{surname}</div>
        <div class="card-row">
            <div>xG: {xg}</div>
            <div>Goal: {goal}</div>
        </div>
        <div class="card-difference">Difference: {diff}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def displayScore(homeScore, awayScore, homeTeam, awayTeam):


    # Somma totale dei punteggi
    somma_totale = homeScore + awayScore

    # Calcolo delle percentuali
    percentuale_squadra_1 = (homeScore / somma_totale) * 100
    percentuale_squadra_2 = (awayScore / somma_totale) * 100

    # HTML e CSS per la barra personalizzata
    st.markdown("""
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

    barra_html = f"""
    
    
    <div class="progress" style="height: 30px;">
        <div class="progress-bar bg-success" role="progressbar" style="font-size: 15px;width:{percentuale_squadra_1}%">
            {homeScore}
        </div>
        <div class="progress-bar bg-danger" role="progressbar" style="font-size: 15px;width:{percentuale_squadra_2}%">
            {awayScore}
        </div>
    </div>
    """

    # Mostra il risultato in Streamlit
    # st.markdown("### Punteggio tra le due squadre:")
    st.markdown(barra_html, unsafe_allow_html=True)

def displayXg(sxg, mxg):
    # HTML e CSS per la barra personalizzata
    st.markdown("""
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

    barra_html1 = f"""
    
    
    <div class="progress" style="height: 30px;">
        <div class="progress-bar bg-success" role="progressbar" style="font-size: 20px;width:{sxg*100}%">
            {sxg}
        </div>
    </div>
    """

    barra_html2 = f"""
    
    
    <div class="progress" style="height: 30px;">
        <div class="progress-bar bg-danger" role="progressbar" style="font-size: 20px;width:{mxg*100}%">
            {mxg}
        </div>
    </div>
    <br>
    """


    st.write("Sofascore xG:")
    st.markdown(barra_html1, unsafe_allow_html=True)
    st.write("Model xG:")
    st.markdown(barra_html2, unsafe_allow_html=True)

    st.warning("Large differences between the two values may be explained by contextual information (opposition's presence and positioning) that are unavailable to the proposed model.")


st.title("Serie A 2025/26")
st.subheader("Filter for Match and Shot to see the shotmap and the xG differences!")
st.write("Last Update: July 15th, 2026")

# with st.expander("Why does the model underestimate some chances?"):
#     st.write("""
#     The proposed model has been trained on several features (Minute, Body Part, Situation, Shooter/Keeper Quality, Team Elos...),
#              but it does not capture the opposition's presence and distance. Therefore, it underestimates the xG values in certain situations.
#     """)
st.warning("The proposed model has been trained on several features (Minute, Body Part, Situation, Shooter/Keeper Quality, Team Elos...), " \
"but it does not capture the opposition's presence and positioning. Therefore, it underestimates the xG values in certain situations.")



# optionMenu1 = option_menu("Pick a League", ["Serie A", "Premier League", "La Liga", "Bundesliga", "Ligue 1"],
#     icons=['1-circle', '2-circle', '3-circle', '4-circle', '5-circle'],menu_icon="trophy-fill",
#     default_index=0, orientation="horizontal"
# )

optionMenu1 = "Serie A"

optionMenu2 = option_menu(None, ["Shots Stats", "Player Stats"],
    icons=['1-circle', '2-circle'], 
    default_index=0, orientation="horizontal"
)
if optionMenu2 == "Shots Stats":
    showShots()
elif optionMenu2 == "Player Stats":
    showPlayers()