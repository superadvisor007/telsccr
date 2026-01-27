# Contributing to TelegramSoccer

Thank you for your interest in contributing! This project aims to be in the top 1% of soccer betting systems through rigorous engineering and analytical practices.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/telegramsoccer.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `make test`
6. Run linters: `make lint`
7. Commit: `git commit -m "Add: your feature description"`
8. Push: `git push origin feature/your-feature-name`
9. Open a Pull Request

## Code Standards

### Python Style
- Follow PEP 8 (enforced by `black`, `isort`, `flake8`)
- Use type hints for all function signatures
- Docstrings for all modules, classes, and public functions
- Maximum line length: 100 characters

### Testing
- Maintain >80% code coverage
- Write unit tests for all new features
- Integration tests for API interactions
- Use pytest fixtures for reusable test data

### Documentation
- Update README.md for user-facing changes
- Update `.github/copilot-instructions.md` for architectural changes
- Add inline comments for complex algorithms
- Document all API endpoints and data schemas

## Areas for Contribution

### High Priority
- **Historical Data Collection**: Build scraper for past match results to train models
- **Model Training**: Implement XGBoost training pipeline with backtesting
- **Sentiment Analysis**: Reddit/Twitter scraper for team/league sentiment
- **Advanced Weather Integration**: More stadium locations and condition mapping
- **UI Dashboard**: Web dashboard for performance visualization

### Medium Priority
- **Additional Markets**: Expand to Asian Handicap, Correct Score
- **Live Betting**: Integrate live odds and in-play analysis
- **Multi-Language Support**: Telegram bot i18n
- **Advanced Bankroll**: Kelly Criterion, dynamic staking
- **Alerting**: Discord/Slack integrations

### Always Welcome
- Bug fixes
- Performance optimizations
- Test coverage improvements
- Documentation enhancements
- Code refactoring

## Commit Message Format

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example: `feat: add Redis caching for API responses`

## Pull Request Process

1. Ensure all tests pass and coverage is maintained
2. Update documentation as needed
3. Add a clear description of changes
4. Link related issues
5. Request review from maintainers
6. Address review feedback

## Betting Ethics

All contributions must:
- Emphasize responsible gambling
- Include appropriate disclaimers
- Not encourage excessive risk-taking
- Promote disciplined, analytical approaches

## Questions?

Open an issue with the `question` label or reach out to maintainers.

Thank you for helping make TelegramSoccer world-class! 🚀⚽
