from flask import Flask, render_template, request, jsonify
from nba_api.stats.endpoints import playercareerstats, teamyearbyyearstats, shotchartdetail, commonplayerinfo, playergamelog
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import leagueleaders, teamplayerdashboard
import pandas as pd
import json
import time
from functools import wraps
import numpy as np

app = Flask(__name__)

# Rate limiting decorator to avoid API overload
def rate_limit(calls_per_second=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(1.0 / calls_per_second)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(0.5)  # 1 call every 2 seconds
def safe_api_call(api_function, **kwargs):
    """Wrapper for NBA API calls with error handling"""
    try:
        return api_function(**kwargs)
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

class NBADataService:
    def __init__(self):
        self.players_cache = None
        self.teams_cache = None
    
    def get_players(self):
        if self.players_cache is None:
            self.players_cache = players.get_players()
        return self.players_cache
    
    def get_teams(self):
        if self.teams_cache is None:
            self.teams_cache = teams.get_teams()
        return self.teams_cache
    
    def search_player(self, player_name):
        """Search for a player by name"""
        all_players = self.get_players()
        matches = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
        return matches[:10]  # Return top 10 matches
    
    def get_player_career_stats(self, player_id):
        """Get career stats for a player"""
        career_stats = safe_api_call(playercareerstats.PlayerCareerStats, player_id=player_id)
        if career_stats:
            return career_stats.get_data_frames()[0]
        return pd.DataFrame()
    
    def get_player_info(self, player_id):
        """Get basic player information"""
        player_info = safe_api_call(commonplayerinfo.CommonPlayerInfo, player_id=player_id)
        if player_info:
            return player_info.get_data_frames()[0]
        return pd.DataFrame()
    
    def get_shot_chart_data(self, player_id, season='2023-24'):
        """Get shot chart data for a player"""
        shot_data = safe_api_call(
            shotchartdetail.ShotChartDetail,
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star='Regular Season'
        )
        if shot_data:
            return shot_data.get_data_frames()[0]
        return pd.DataFrame()
    
    def get_team_stats(self, team_id, season='2023-24'):
        """Get team statistics"""
        team_stats = safe_api_call(teamyearbyyearstats.TeamYearByYearStats, team_id=team_id)
        if team_stats:
            return team_stats.get_data_frames()[0]
        return pd.DataFrame()
    
    def compare_players(self, player1_id, player2_id, season='2023-24'):
        """Compare two players' statistics"""
        p1_stats = self.get_player_career_stats(player1_id)
        p2_stats = self.get_player_career_stats(player2_id)
        p1_info = self.get_player_info(player1_id)
        p2_info = self.get_player_info(player2_id)
        
        # Get latest season stats
        if not p1_stats.empty:
            p1_latest = p1_stats.iloc[-1] if len(p1_stats) > 0 else pd.Series()
        else:
            p1_latest = pd.Series()
            
        if not p2_stats.empty:
            p2_latest = p2_stats.iloc[-1] if len(p2_stats) > 0 else pd.Series()
        else:
            p2_latest = pd.Series()
        
        return {
            'player1': {
                'info': p1_info.to_dict('records')[0] if not p1_info.empty else {},
                'stats': p1_latest.to_dict() if not p1_latest.empty else {}
            },
            'player2': {
                'info': p2_info.to_dict('records')[0] if not p2_info.empty else {},
                'stats': p2_latest.to_dict() if not p2_latest.empty else {}
            }
        }

# Initialize the NBA data service
nba_service = NBADataService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/charts')
def charts():
    return render_template('charts.html')

@app.route('/teams')
def teams_page():
    return render_template('teams.html')

@app.route('/api/search_players')
def search_players():
    query = request.args.get('q', '')
    if len(query) < 2:
        return jsonify([])
    
    matches = nba_service.search_player(query)
    return jsonify(matches)

@app.route('/api/player_info/<int:player_id>')
def get_player_info(player_id):
    info = nba_service.get_player_info(player_id)
    if not info.empty:
        return jsonify(info.to_dict('records')[0])
    return jsonify({})

@app.route('/api/compare_players')
def compare_players():
    player1_id = request.args.get('player1_id', type=int)
    player2_id = request.args.get('player2_id', type=int)
    season = request.args.get('season', '2023-24')
    
    if not player1_id or not player2_id:
        return jsonify({'error': 'Both player IDs are required'})
    
    comparison = nba_service.compare_players(player1_id, player2_id, season)
    return jsonify(comparison)

@app.route('/api/shot_chart/<int:player_id>')
def get_shot_chart(player_id):
    season = request.args.get('season', '2023-24')
    shot_data = nba_service.get_shot_chart_data(player_id, season)
    
    if not shot_data.empty:
        # Convert to format suitable for visualization
        shots = []
        for _, shot in shot_data.iterrows():
            shots.append({
                'x': shot.get('LOC_X', 0),
                'y': shot.get('LOC_Y', 0),
                'made': shot.get('SHOT_MADE_FLAG', 0),
                'zone': shot.get('SHOT_ZONE_BASIC', 'Unknown'),
                'distance': shot.get('SHOT_DISTANCE', 0)
            })
        return jsonify(shots)
    
    return jsonify([])

@app.route('/api/teams')
def get_teams():
    teams_list = nba_service.get_teams()
    return jsonify(teams_list)

@app.route('/api/team_stats/<int:team_id>')
def get_team_stats(team_id):
    season = request.args.get('season', '2023-24')
    stats = nba_service.get_team_stats(team_id, season)
    
    if not stats.empty:
        return jsonify(stats.to_dict('records'))
    return jsonify([])

# Sample data generator for demo purposes when API is unavailable
def generate_sample_data():
    """Generate sample data for demo purposes"""
    sample_players = [
        {'id': 2544, 'full_name': 'LeBron James', 'first_name': 'LeBron', 'last_name': 'James', 'is_active': True},
        {'id': 201939, 'full_name': 'Stephen Curry', 'first_name': 'Stephen', 'last_name': 'Curry', 'is_active': True},
        {'id': 201142, 'full_name': 'Kevin Durant', 'first_name': 'Kevin', 'last_name': 'Durant', 'is_active': True},
        {'id': 201566, 'full_name': 'Russell Westbrook', 'first_name': 'Russell', 'last_name': 'Westbrook', 'is_active': True},
        {'id': 203076, 'full_name': 'Anthony Davis', 'first_name': 'Anthony', 'last_name': 'Davis', 'is_active': True}
    ]
    return sample_players

@app.route('/api/sample_comparison')
def sample_comparison():
    """Provide sample comparison data for demo"""
    return jsonify({
        'player1': {
            'info': {
                'DISPLAY_FIRST_LAST': 'LeBron James',
                'TEAM_CITY': 'Los Angeles',
                'TEAM_NAME': 'Lakers',
                'POSITION': 'F',
                'HEIGHT': '6-9',
                'WEIGHT': '250'
            },
            'stats': {
                'PTS': 25.7,
                'AST': 8.3,
                'REB': 7.3,
                'FG_PCT': 0.540,
                'FT_PCT': 0.741,
                'FG3_PCT': 0.410,
                'STL': 1.3,
                'BLK': 0.5,
                'TOV': 3.5,
                'MIN': 35.3,
                'GP': 71
            }
        },
        'player2': {
            'info': {
                'DISPLAY_FIRST_LAST': 'Kevin Durant',
                'TEAM_CITY': 'Phoenix',
                'TEAM_NAME': 'Suns',
                'POSITION': 'F',
                'HEIGHT': '6-10',
                'WEIGHT': '240'
            },
            'stats': {
                'PTS': 27.1,
                'AST': 5.0,
                'REB': 6.6,
                'FG_PCT': 0.523,
                'FT_PCT': 0.856,
                'FG3_PCT': 0.413,
                'STL': 0.9,
                'BLK': 1.2,
                'TOV': 3.3,
                'MIN': 37.2,
                'GP': 75
            }
        }
    })

@app.route('/api/sample_shot_chart')
def sample_shot_chart():
    """Generate sample shot chart data"""
    np.random.seed(42)  # For consistent demo data
    shots = []
    
    # Generate realistic shot locations
    for i in range(200):
        # Various shot zones
        zone_type = np.random.choice(['paint', 'mid_range', 'three_point'], p=[0.4, 0.3, 0.3])
        
        if zone_type == 'paint':
            x = np.random.normal(0, 50)
            y = np.random.uniform(-80, 80)
            made = np.random.choice([0, 1], p=[0.35, 0.65])  # Higher success rate in paint
        elif zone_type == 'mid_range':
            x = np.random.uniform(-150, 150)
            y = np.random.uniform(-200, 200)
            made = np.random.choice([0, 1], p=[0.55, 0.45])  # Lower mid-range success
        else:  # three_point
            angle = np.random.uniform(0, np.pi)
            distance = np.random.uniform(238, 280)  # 3-point line distance
            x = distance * np.cos(angle)
            y = distance * np.sin(angle) - 50
            made = np.random.choice([0, 1], p=[0.62, 0.38])  # 3-point success rate
        
        shots.append({
            'x': int(x),
            'y': int(y),
            'made': made,
            'zone': zone_type,
            'distance': int(np.sqrt(x**2 + y**2) / 10)
        })
    
    return jsonify(shots)

if __name__ == '__main__':
    app.run(debug=True)