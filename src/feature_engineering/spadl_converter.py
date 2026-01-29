"""
🔄 SPADL Converter
==================
Convert raw event data to SPADL (Soccer Player Action Description Language) format.

SPADL is a standardized format for representing soccer actions, developed by
the KU Leuven ML group (socceraction library).

Source: https://github.com/ML-KULeuven/socceraction

Key Features:
- Unified action representation
- Compatible with VAEP, xT, and other valuation frameworks
- Standardized field coordinates (0-100 x 0-68)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# SPADL Action Types
ACTION_TYPES = [
    'pass',
    'cross',
    'throw_in',
    'freekick_crossed',
    'freekick_short',
    'corner_crossed',
    'corner_short',
    'take_on',
    'foul',
    'tackle',
    'interception',
    'shot',
    'shot_penalty',
    'shot_freekick',
    'keeper_save',
    'keeper_claim',
    'keeper_punch',
    'keeper_pick_up',
    'clearance',
    'bad_touch',
    'non_action',
    'dribble',
    'goalkick',
]

# SPADL Body Parts
BODY_PARTS = ['foot', 'head', 'other', 'head/other']

# SPADL Results
RESULTS = ['fail', 'success', 'offside', 'owngoal', 'yellow_card', 'red_card']


@dataclass
class SPADLAction:
    """A single SPADL action."""
    game_id: int
    period_id: int
    time_seconds: float
    team_id: int
    player_id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    action_type: str
    result: str
    body_part: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'game_id': self.game_id,
            'period_id': self.period_id,
            'time_seconds': self.time_seconds,
            'team_id': self.team_id,
            'player_id': self.player_id,
            'start_x': self.start_x,
            'start_y': self.start_y,
            'end_x': self.end_x,
            'end_y': self.end_y,
            'type_name': self.action_type,
            'result_name': self.result,
            'bodypart_name': self.body_part,
        }


class SPADLConverter:
    """
    🔄 SPADL Format Converter
    
    Converts various event data formats to standardized SPADL.
    
    Supported Input Formats:
    - StatsBomb events
    - Wyscout events
    - Opta events
    
    Output: Pandas DataFrame in SPADL format
    
    Example:
        converter = SPADLConverter()
        
        # Convert StatsBomb events
        spadl_df = converter.convert_statsbomb(events_df)
        
        # Get action features
        features = converter.get_action_features(spadl_df)
    """
    
    # Field dimensions for normalization
    FIELD_LENGTH = 105.0
    FIELD_WIDTH = 68.0
    
    # StatsBomb uses 120x80 coordinate system
    STATSBOMB_LENGTH = 120.0
    STATSBOMB_WIDTH = 80.0
    
    def __init__(self):
        self._type_mapping = self._build_type_mapping()
    
    def _build_type_mapping(self) -> Dict[str, str]:
        """Build mapping from StatsBomb event types to SPADL actions."""
        return {
            'Pass': 'pass',
            'Ball Receipt*': 'non_action',
            'Carry': 'dribble',
            'Shot': 'shot',
            'Pressure': 'non_action',
            'Duel': 'tackle',
            'Interception': 'interception',
            'Block': 'interception',
            'Clearance': 'clearance',
            'Dribble': 'take_on',
            'Foul Committed': 'foul',
            'Foul Won': 'non_action',
            'Ball Recovery': 'interception',
            'Miscontrol': 'bad_touch',
            'Dispossessed': 'non_action',
            'Goal Keeper': 'keeper_save',
            'Tactical Shift': 'non_action',
            'Injury Stoppage': 'non_action',
            'Substitution': 'non_action',
            'Half Start': 'non_action',
            'Half End': 'non_action',
            'Starting XI': 'non_action',
            'Referee Ball-Drop': 'non_action',
            'Shield': 'non_action',
            '50/50': 'tackle',
            'Error': 'bad_touch',
            'Offside': 'non_action',
            'Own Goal Against': 'non_action',
            'Own Goal For': 'shot',
            'Player Off': 'non_action',
            'Player On': 'non_action',
        }
    
    def convert_statsbomb(
        self,
        events: pd.DataFrame,
        game_id: int = None
    ) -> pd.DataFrame:
        """
        Convert StatsBomb events to SPADL format.
        
        Args:
            events: StatsBomb events DataFrame
            game_id: Optional game identifier
        
        Returns:
            SPADL-formatted DataFrame
        """
        actions = []
        
        for _, event in events.iterrows():
            action = self._convert_statsbomb_event(event, game_id)
            if action:
                actions.append(action.to_dict())
        
        df = pd.DataFrame(actions)
        
        # Add action type and result IDs
        if len(df) > 0:
            df['type_id'] = df['type_name'].map(
                {t: i for i, t in enumerate(ACTION_TYPES)}
            ).fillna(-1).astype(int)
            
            df['result_id'] = df['result_name'].map(
                {r: i for i, r in enumerate(RESULTS)}
            ).fillna(0).astype(int)
            
            df['bodypart_id'] = df['bodypart_name'].map(
                {b: i for i, b in enumerate(BODY_PARTS)}
            ).fillna(0).astype(int)
        
        return df
    
    def _convert_statsbomb_event(
        self,
        event: pd.Series,
        game_id: int = None
    ) -> Optional[SPADLAction]:
        """Convert a single StatsBomb event to SPADL action."""
        
        event_type = event.get('type', '')
        action_type = self._type_mapping.get(event_type, 'non_action')
        
        # Skip non-actions
        if action_type == 'non_action':
            return None
        
        # Get coordinates
        start_x = event.get('location_x', 0) or 0
        start_y = event.get('location_y', 0) or 0
        
        # Normalize to standard field (0-105, 0-68)
        start_x = start_x / self.STATSBOMB_LENGTH * self.FIELD_LENGTH
        start_y = start_y / self.STATSBOMB_WIDTH * self.FIELD_WIDTH
        
        # Get end location
        end_x, end_y = start_x, start_y
        
        if event_type == 'Pass':
            end_x = event.get('pass_end_x', start_x) or start_x
            end_y = event.get('pass_end_y', start_y) or start_y
            end_x = end_x / self.STATSBOMB_LENGTH * self.FIELD_LENGTH
            end_y = end_y / self.STATSBOMB_WIDTH * self.FIELD_WIDTH
            
            # Determine pass type
            pass_type = event.get('pass_type')
            if pass_type == 'Corner':
                action_type = 'corner_crossed'
            elif pass_type == 'Free Kick':
                action_type = 'freekick_short'
            elif pass_type == 'Throw-in':
                action_type = 'throw_in'
            elif event.get('pass_cross'):
                action_type = 'cross'
        
        elif event_type == 'Shot':
            end_x = event.get('shot_end_x', 105) or 105
            end_y = event.get('shot_end_y', 34) or 34
            
            shot_type = event.get('shot_type')
            if shot_type == 'Penalty':
                action_type = 'shot_penalty'
            elif shot_type == 'Free Kick':
                action_type = 'shot_freekick'
        
        elif event_type == 'Carry':
            end_x = event.get('carry_end_x', start_x) or start_x
            end_y = event.get('carry_end_y', start_y) or start_y
            end_x = end_x / self.STATSBOMB_LENGTH * self.FIELD_LENGTH
            end_y = end_y / self.STATSBOMB_WIDTH * self.FIELD_WIDTH
        
        # Determine result
        result = 'success'
        
        if event_type == 'Pass':
            if event.get('pass_outcome') and event.get('pass_outcome') != 'Complete':
                result = 'fail'
        elif event_type == 'Shot':
            outcome = event.get('shot_outcome', '')
            if outcome in ['Goal', 'Saved']:
                result = 'success'
            else:
                result = 'fail'
        elif event_type == 'Dribble':
            if event.get('dribble_outcome') == 'Incomplete':
                result = 'fail'
        
        # Determine body part
        body_part = 'foot'
        
        if event_type == 'Pass':
            bp = event.get('pass_body_part', '')
            if 'Head' in str(bp):
                body_part = 'head'
            elif bp and 'Foot' not in str(bp):
                body_part = 'other'
        elif event_type == 'Shot':
            bp = event.get('shot_body_part', '')
            if 'Head' in str(bp):
                body_part = 'head'
            elif bp and 'Foot' not in str(bp):
                body_part = 'other'
        
        # Get time
        minute = event.get('minute', 0) or 0
        second = event.get('second', 0) or 0
        time_seconds = minute * 60 + second
        
        return SPADLAction(
            game_id=game_id or 0,
            period_id=event.get('period', 1) or 1,
            time_seconds=time_seconds,
            team_id=event.get('team_id', 0) or 0,
            player_id=event.get('player_id', 0) or 0,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            action_type=action_type,
            result=result,
            body_part=body_part
        )
    
    def get_action_features(self, spadl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from SPADL actions.
        
        Features:
        - Spatial features (zones, distance to goal)
        - Action context (previous action, possession length)
        - Action quality metrics
        """
        if len(spadl_df) == 0:
            return pd.DataFrame()
        
        features = spadl_df.copy()
        
        # Spatial features
        features['start_dist_to_goal'] = self._distance_to_goal(
            features['start_x'], features['start_y']
        )
        features['end_dist_to_goal'] = self._distance_to_goal(
            features['end_x'], features['end_y']
        )
        features['move_dist'] = np.sqrt(
            (features['end_x'] - features['start_x'])**2 +
            (features['end_y'] - features['start_y'])**2
        )
        
        # Zone features
        features['start_zone'] = self._get_zone(
            features['start_x'], features['start_y']
        )
        features['end_zone'] = self._get_zone(
            features['end_x'], features['end_y']
        )
        
        # Angle to goal
        features['angle_to_goal'] = self._angle_to_goal(
            features['start_x'], features['start_y']
        )
        
        # Action progression (how much closer to goal)
        features['progression'] = features['start_dist_to_goal'] - features['end_dist_to_goal']
        
        # In penalty area
        features['in_penalty_area'] = (
            (features['end_x'] > 88.5) &  # Last 16.5m
            (features['end_y'] > 13.84) &
            (features['end_y'] < 54.16)
        ).astype(int)
        
        return features
    
    def _distance_to_goal(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Calculate distance to center of goal."""
        goal_x = 105  # Right side goal
        goal_y = 34   # Center
        return np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
    
    def _angle_to_goal(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Calculate angle to goal from position."""
        goal_x = 105
        goal_y = 34
        return np.degrees(np.arctan2(goal_y - y, goal_x - x))
    
    def _get_zone(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Divide field into zones (1-18)."""
        # 3 vertical x 6 horizontal zones
        x_zone = pd.cut(x, bins=[0, 35, 70, 105], labels=[1, 2, 3]).astype(int)
        y_zone = pd.cut(y, bins=[0, 11.3, 22.6, 34, 45.3, 56.6, 68], labels=[1, 2, 3, 4, 5, 6]).astype(int)
        return (x_zone - 1) * 6 + y_zone
    
    def compute_possession_chains(self, spadl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify possession chains in SPADL actions.
        
        A possession chain ends when:
        - Ball goes out of play
        - Team loses possession
        - Goal is scored
        """
        df = spadl_df.copy()
        
        # Detect possession changes
        df['team_change'] = df['team_id'] != df['team_id'].shift(1)
        df['period_change'] = df['period_id'] != df['period_id'].shift(1)
        
        # Failed actions that likely end possession
        df['possession_end'] = (
            (df['result_name'] == 'fail') & 
            (df['type_name'].isin(['pass', 'cross', 'shot', 'take_on', 'bad_touch']))
        )
        
        # Mark chain starts
        df['chain_start'] = df['team_change'] | df['period_change'] | df['possession_end'].shift(1).fillna(False)
        
        # Create chain IDs
        df['chain_id'] = df['chain_start'].cumsum()
        
        # Calculate chain length
        chain_lengths = df.groupby('chain_id').size().reset_index(name='chain_length')
        df = df.merge(chain_lengths, on='chain_id')
        
        return df
