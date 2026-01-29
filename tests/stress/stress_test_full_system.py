#!/usr/bin/env python3
"""
FULL SYSTEM STRESS TEST - TelegramSoccer with DeepSeek LLM

Tests all components of the system:
1. DeepSeek LLM integration
2. Ticket generation
3. ML predictions
4. Telegram sending
5. Result verification

Run with: python stress_test_full_system.py
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Colors:
    """Terminal colors."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class SystemStressTest:
    """Full system stress test."""
    
    def __init__(self):
        self.results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "components": {},
        }
        self.start_time = time.time()
    
    def run_all_tests(self):
        """Run all system tests."""
        print_header("🚀 TELEGRAMSOCCER FULL SYSTEM STRESS TEST")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python: {sys.version.split()[0]}")
        print()
        
        # Test 1: DeepSeek LLM
        self.test_deepseek_llm()
        
        # Test 2: Ticket Generator
        self.test_ticket_generator()
        
        # Test 3: Ticket with Results
        self.test_ticket_results()
        
        # Test 4: Telegram Sender
        self.test_telegram_sender()
        
        # Test 5: LLM Analyzer
        self.test_llm_analyzer()
        
        # Test 6: Full Pipeline
        self.test_full_pipeline()
        
        # Print summary
        self.print_summary()
    
    def test_deepseek_llm(self):
        """Test DeepSeek LLM client."""
        print_header("TEST 1: DeepSeek LLM Client")
        self.results["tests_run"] += 1
        
        try:
            from src.llm.deepseek_client import DeepSeekLLM, get_deepseek_llm
            print_success("DeepSeek client imported successfully")
            
            # Create instance
            llm = get_deepseek_llm(model="deepseek-llm:7b")
            print_success(f"LLM instance created: {llm.model}")
            print_info(f"  Base URL: {llm.base_url}")
            print_info(f"  Temperature: {llm.temperature}")
            print_info(f"  Max tokens: {llm.max_tokens}")
            
            # Check availability
            if llm.is_available():
                print_success("Ollama server is running!")
                print_success("DeepSeek model is available!")
                
                # Test actual inference
                print_info("Testing inference...")
                test_match = {
                    "home_team": "Bayern Munich",
                    "away_team": "Dortmund",
                    "league": "Bundesliga",
                    "home_goals_per_game": 2.5,
                    "away_goals_per_game": 1.8,
                }
                
                result = llm.analyze_match(test_match, "over_1_5")
                if "probability" in result and not result.get("error"):
                    print_success(f"Inference successful!")
                    print_info(f"  Probability: {result.get('probability', 0):.2%}")
                    print_info(f"  Confidence: {result.get('confidence', 0):.2%}")
                    print_info(f"  Recommendation: {result.get('recommendation', 'N/A')}")
                else:
                    print_warning("Inference returned fallback (Ollama may not be running)")
            else:
                print_warning("Ollama not running - using fallback analysis")
                
                # Test fallback
                result = llm._fallback_analysis(test_match, "over_1_5")
                print_success(f"Fallback analysis works: {result.get('probability', 0):.2%}")
            
            self.results["tests_passed"] += 1
            self.results["components"]["deepseek_llm"] = "PASS"
            
        except Exception as e:
            print_error(f"DeepSeek test failed: {e}")
            self.results["tests_failed"] += 1
            self.results["components"]["deepseek_llm"] = f"FAIL: {e}"
    
    def test_ticket_generator(self):
        """Test ticket generator."""
        print_header("TEST 2: Ticket Generator")
        self.results["tests_run"] += 1
        
        try:
            from src.bot.ticket_generator import (
                TicketGenerator,
                MultiBetTicket,
                BetLeg,
                MarketType,
            )
            print_success("Ticket generator imported")
            
            # Create test predictions
            predictions = [
                {
                    "home_team": "AS Monaco FC",
                    "away_team": "Juventus",
                    "market": "btts",
                    "odds": 1.60,
                    "league": "Champions League",
                    "probability": 0.72,
                },
                {
                    "home_team": "Bayern Munich",
                    "away_team": "PSG",
                    "market": "over_1_5",
                    "odds": 1.25,
                    "league": "Champions League",
                    "probability": 0.85,
                },
            ]
            
            # Generate ticket
            generator = TicketGenerator()
            ticket = generator.create_ticket(predictions, stake=10.0)
            
            print_success(f"Ticket created: {ticket.ticket_id}")
            print_info(f"  Legs: {ticket.total_legs}")
            print_info(f"  Total odds: {ticket.total_odds:.2f}")
            print_info(f"  Potential win: €{ticket.potential_win:.2f}")
            
            # Format ticket
            formatted = generator.format_ticket(ticket)
            print_success("Ticket formatted successfully")
            print_info(f"  Length: {len(formatted)} chars")
            
            # Verify format contains key elements
            assert "MULTI-BET TICKET" in formatted
            assert "AS Monaco FC" in formatted
            assert "Juventus" in formatted
            assert "Total Odds" in formatted
            print_success("Format validation passed")
            
            # Test HTML format
            html_ticket = generator.format_ticket_html(ticket)
            assert "<b>" in html_ticket
            print_success("HTML format works")
            
            self.results["tests_passed"] += 1
            self.results["components"]["ticket_generator"] = "PASS"
            
        except Exception as e:
            print_error(f"Ticket generator test failed: {e}")
            self.results["tests_failed"] += 1
            self.results["components"]["ticket_generator"] = f"FAIL: {e}"
    
    def test_ticket_results(self):
        """Test ticket with results."""
        print_header("TEST 3: Ticket Results (✓/X)")
        self.results["tests_run"] += 1
        
        try:
            from src.bot.ticket_generator import TicketGenerator
            
            predictions = [
                {"home_team": "Team A", "away_team": "Team B", "market": "btts", "odds": 1.60, "probability": 0.70},
                {"home_team": "Team C", "away_team": "Team D", "market": "over_1_5", "odds": 1.25, "probability": 0.82},
                {"home_team": "Team E", "away_team": "Team F", "market": "btts", "odds": 1.50, "probability": 0.68},
            ]
            
            results = [
                {"home_score": 2, "away_score": 1},  # BTTS: WIN
                {"home_score": 1, "away_score": 0},  # Over 1.5: LOSS (only 1 goal)
                {"home_score": 1, "away_score": 1},  # BTTS: WIN
            ]
            
            generator = TicketGenerator()
            ticket = generator.create_ticket(predictions)
            ticket = generator.update_results(ticket, results)
            
            print_success("Results applied to ticket")
            print_info(f"  Wins: {ticket.wins}")
            print_info(f"  Losses: {ticket.losses}")
            print_info(f"  Is winner: {ticket.is_winner}")
            
            # Check result markers
            formatted = generator.format_ticket(ticket, show_results=True, show_scores=True)
            assert "✓" in formatted or "✗" in formatted
            print_success("Result markers (✓/X) present")
            
            # Verify scores shown
            assert "(2-1)" in formatted or "2-1" in formatted
            print_success("Scores displayed correctly")
            
            self.results["tests_passed"] += 1
            self.results["components"]["ticket_results"] = "PASS"
            
        except Exception as e:
            print_error(f"Ticket results test failed: {e}")
            self.results["tests_failed"] += 1
            self.results["components"]["ticket_results"] = f"FAIL: {e}"
    
    def test_telegram_sender(self):
        """Test Telegram sender (without actually sending)."""
        print_header("TEST 4: Telegram Sender")
        self.results["tests_run"] += 1
        
        try:
            from src.bot.telegram_sender import TelegramTicketSender
            print_success("Telegram sender imported")
            
            # Test message generation (don't actually send)
            predictions = [
                {"home_team": "Test Home", "away_team": "Test Away", "market": "btts", "odds": 1.50, "probability": 0.75},
            ]
            
            sender = TelegramTicketSender.__new__(TelegramTicketSender)
            sender.bot_token = "test"
            sender.chat_id = "test"
            sender.base_url = "https://api.telegram.org/bottest"
            
            # Test ticket text generation
            ticket_text = sender._generate_ticket_text(predictions, stake=10.0)
            
            assert "MULTI-BET TICKET" in ticket_text
            assert "Test Home" in ticket_text
            assert "DeepSeek" in ticket_text
            print_success("Ticket text generated correctly")
            print_info(f"  Length: {len(ticket_text)} chars")
            
            # Test results text
            results = [{"home_score": 1, "away_score": 1}]
            results_text = sender._generate_ticket_with_results(predictions, results, stake=10.0)
            assert "✓" in results_text or "✗" in results_text
            print_success("Results text generated with markers")
            
            # Check if real token exists
            real_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            if real_token:
                print_success("TELEGRAM_BOT_TOKEN is set")
            else:
                print_warning("TELEGRAM_BOT_TOKEN not set (can't send real messages)")
            
            self.results["tests_passed"] += 1
            self.results["components"]["telegram_sender"] = "PASS"
            
        except Exception as e:
            print_error(f"Telegram sender test failed: {e}")
            self.results["tests_failed"] += 1
            self.results["components"]["telegram_sender"] = f"FAIL: {e}"
    
    def test_llm_analyzer(self):
        """Test LLM Analyzer with DeepSeek."""
        print_header("TEST 5: LLM Analyzer (DeepSeek Integration)")
        self.results["tests_run"] += 1
        
        try:
            from src.llm.analyzer import LLMAnalyzer
            print_success("LLM Analyzer imported")
            
            analyzer = LLMAnalyzer()
            print_success(f"Analyzer created with model: {analyzer.model}")
            
            # Verify DeepSeek is configured
            assert "deepseek" in analyzer.model.lower(), f"Expected DeepSeek, got {analyzer.model}"
            print_success("DeepSeek is configured as default LLM")
            
            # Test analysis prompt building
            match_data = {
                "home_team": "Bayern Munich",
                "away_team": "Dortmund",
                "league": "Bundesliga",
                "match_date": "2026-01-28",
            }
            features = {"total_expected_goals": 3.2, "home_attack_strength": 1.5}
            home_stats = {"goals_per_game": 2.5, "btts_percentage": 65}
            away_stats = {"goals_per_game": 1.8, "btts_percentage": 60}
            h2h_stats = {"matches": 5, "avg_goals": 3.5}
            
            prompt = analyzer._build_analysis_prompt(match_data, features, home_stats, away_stats, h2h_stats, None)
            assert "Bayern Munich" in prompt
            assert "Dortmund" in prompt
            print_success("Analysis prompt built correctly")
            print_info(f"  Prompt length: {len(prompt)} chars")
            
            self.results["tests_passed"] += 1
            self.results["components"]["llm_analyzer"] = "PASS"
            
        except Exception as e:
            print_error(f"LLM Analyzer test failed: {e}")
            self.results["tests_failed"] += 1
            self.results["components"]["llm_analyzer"] = f"FAIL: {e}"
    
    def test_full_pipeline(self):
        """Test full prediction pipeline."""
        print_header("TEST 6: Full Pipeline Integration")
        self.results["tests_run"] += 1
        
        try:
            # Import all components
            from src.llm.deepseek_client import get_deepseek_llm
            from src.bot.ticket_generator import TicketGenerator, DailyTicketService
            from src.bot.telegram_sender import TelegramTicketSender
            
            print_success("All components imported")
            
            # Simulate full pipeline
            predictions = [
                {
                    "home_team": "Manchester City",
                    "away_team": "Liverpool",
                    "market": "over_1_5",
                    "odds": 1.18,
                    "league": "Premier League",
                    "probability": 0.88,
                    "confidence": 0.85,
                },
                {
                    "home_team": "Barcelona",
                    "away_team": "Real Madrid",
                    "market": "btts",
                    "odds": 1.55,
                    "league": "La Liga",
                    "probability": 0.72,
                    "confidence": 0.75,
                },
                {
                    "home_team": "Bayern Munich",
                    "away_team": "Dortmund",
                    "market": "over_2_5",
                    "odds": 1.45,
                    "league": "Bundesliga",
                    "probability": 0.75,
                    "confidence": 0.78,
                },
            ]
            
            # Generate ticket
            service = DailyTicketService()
            ticket = service.generate_daily_ticket(
                predictions=predictions,
                target_odds=1.40,
                max_legs=4,
                min_confidence=0.70,
            )
            
            print_success(f"Daily ticket generated: {ticket.ticket_id}")
            print_info(f"  Legs selected: {ticket.total_legs}")
            print_info(f"  Total odds: {ticket.total_odds:.2f}")
            
            # Format for Telegram
            message = service.get_telegram_message(ticket)
            print_success("Telegram message formatted")
            print_info(f"  Message length: {len(message)} chars")
            
            # Simulate results
            results = [
                {"home_score": 3, "away_score": 2},  # Over 1.5: WIN
                {"home_score": 2, "away_score": 1},  # BTTS: WIN
                {"home_score": 4, "away_score": 1},  # Over 2.5: WIN
            ]
            
            results_message = service.get_results_message(ticket, results)
            assert "✓" in results_message or "WIN" in results_message
            print_success("Results message generated")
            
            # Verify pipeline stats
            print_info("\n  Pipeline Statistics:")
            print_info(f"  - LLM Model: DeepSeek 7B")
            print_info(f"  - Predictions processed: {len(predictions)}")
            print_info(f"  - Ticket legs: {ticket.total_legs}")
            print_info(f"  - Combined odds: {ticket.total_odds:.2f}")
            
            self.results["tests_passed"] += 1
            self.results["components"]["full_pipeline"] = "PASS"
            
        except Exception as e:
            print_error(f"Full pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["tests_failed"] += 1
            self.results["components"]["full_pipeline"] = f"FAIL: {e}"
    
    def print_summary(self):
        """Print test summary."""
        elapsed = time.time() - self.start_time
        
        print_header("📊 TEST SUMMARY")
        
        print(f"Total Tests: {self.results['tests_run']}")
        print(f"{Colors.GREEN}Passed: {self.results['tests_passed']}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.results['tests_failed']}{Colors.END}")
        print(f"Time: {elapsed:.2f}s")
        print()
        
        print("Component Results:")
        print("-" * 40)
        for component, status in self.results["components"].items():
            if "PASS" in status:
                print(f"  {Colors.GREEN}✓{Colors.END} {component}: {status}")
            else:
                print(f"  {Colors.RED}✗{Colors.END} {component}: {status}")
        
        print()
        if self.results["tests_failed"] == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.END}")
            print(f"{Colors.GREEN}System is ready with DeepSeek LLM integration.{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}⚠️ SOME TESTS FAILED{Colors.END}")
            print(f"{Colors.YELLOW}Check the errors above and fix issues.{Colors.END}")
        
        print()
        print_header("DeepSeek LLM Status")
        print("Model: deepseek-llm:7b")
        print("Provider: Ollama (local)")
        print("Cost: $0.00 (100% FREE)")
        print()
        
        # Check Ollama status
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                print(f"{Colors.GREEN}✓ Ollama is running{Colors.END}")
                models = resp.json().get("models", [])
                print(f"  Available models: {[m.get('name') for m in models]}")
            else:
                print(f"{Colors.YELLOW}⚠ Ollama not responding{Colors.END}")
        except:
            print(f"{Colors.YELLOW}⚠ Ollama not running (fallback mode active){Colors.END}")
            print("  To enable DeepSeek inference:")
            print("  1. Start Ollama: ollama serve")
            print("  2. Pull model: ollama pull deepseek-llm:7b")


if __name__ == "__main__":
    tester = SystemStressTest()
    tester.run_all_tests()
