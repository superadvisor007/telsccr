"""
💰 BETTING MODULE
================
Multi-bet ticket building and betting logic.

Components:
- multibet_ticket_builder: Enhanced ticket builder with reasoning integration
- ticket_builder: Legacy ticket builder
- value_detector: Value bet identification
- bankroll_manager: Kelly criterion and staking

Constraints (Battle-Tested):
- Single leg odds: 1.40 - 1.70
- Target total odds: ~10x
- Min confidence: 65%
- Min edge: 5%
"""

from .multibet_ticket_builder import (
    TicketConfig,
    EnhancedBetLeg,
    EnhancedTicket,
    MultiBetTicketBuilder
)

__all__ = [
    'TicketConfig',
    'EnhancedBetLeg',
    'EnhancedTicket',
    'MultiBetTicketBuilder'
]
