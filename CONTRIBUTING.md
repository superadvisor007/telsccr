# Contributing to Swiss Soccer Tips Bot

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## How to Contribute

### Reporting Bugs

1. Check if the issue already exists
2. Use the bug report template
3. Include:
   - Description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version)
   - Logs if applicable

### Suggesting Features

1. Open an issue with the feature request template
2. Describe:
   - The problem it solves
   - Proposed solution
   - Alternative solutions considered
   - Additional context

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/telegramsoccer.git
   cd telegramsoccer
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style (see below)
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   python test.py
   python -m py_compile src/*.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub

## Code Style

### Python Style Guide

Follow PEP 8 with these specifics:

- **Indentation**: 4 spaces
- **Line length**: Max 100 characters (120 for comments)
- **Docstrings**: Use Google style
- **Type hints**: Use where applicable

Example:
```python
async def create_prediction(
    self,
    match_id: int,
    prediction: str,
    confidence: float
) -> int:
    """Create a new prediction.
    
    Args:
        match_id: Unique match identifier
        prediction: Prediction outcome (home_win|away_win|draw)
        confidence: Confidence score (0.0-1.0)
        
    Returns:
        Prediction ID
        
    Raises:
        ValueError: If confidence is out of range
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1")
    # Implementation
    return prediction_id
```

### Naming Conventions

- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add TWINT payment support
fix: resolve database connection timeout
docs: update deployment guide
refactor: simplify prediction engine logic
```

## Project Structure

```
telegramsoccer/
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # Main orchestrator
│   ├── bot.py             # Telegram bot
│   ├── database.py        # Database models
│   ├── api_football.py    # API client
│   ├── prediction_engine.py  # AI predictions
│   ├── payment_handler.py    # Stripe integration
│   └── webhook_server.py     # Webhook server
├── .github/
│   └── workflows/         # GitHub Actions
├── data/                  # SQLite database (gitignored)
├── logs/                  # Log files (gitignored)
├── test.py               # Test suite
├── run.py                # CLI runner
└── README.md
```

## Development Setup

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/telegramsoccer.git
   cd telegramsoccer
   pip install -r requirements.txt
   cp .env.example .env
   ```

2. **Edit .env with test credentials**
   - Use test API keys
   - Use test Stripe keys (sk_test_...)
   - Create a test Telegram bot

3. **Run tests**
   ```bash
   python test.py
   ```

4. **Test locally**
   ```bash
   python run.py bot
   ```

## Testing Guidelines

### Writing Tests

Add tests for new features in `test.py`:

```python
async def test_your_feature():
    """Test your new feature."""
    # Setup
    db = Database("./data/test_bot.db")
    await db.connect()
    
    # Test
    result = await your_feature()
    assert result is not None
    
    # Cleanup
    await db.close()
```

### Manual Testing

Before submitting PR:

1. ✅ Bot starts without errors
2. ✅ Commands respond correctly
3. ✅ Database operations work
4. ✅ No syntax errors (`python -m py_compile src/*.py`)
5. ✅ Tests pass (`python test.py`)

## Documentation

Update documentation when:

- Adding new features
- Changing configuration
- Modifying API
- Adding dependencies

Files to update:
- `README.md` - Main documentation
- `DEPLOYMENT.md` - Deployment instructions
- Docstrings in code

## API Changes

If changing APIs:

1. Maintain backwards compatibility when possible
2. Document breaking changes clearly
3. Update version number appropriately
4. Add migration guide if needed

## Database Changes

If modifying database schema:

1. Add migration script
2. Test with existing data
3. Document changes
4. Increment schema version

## Adding Dependencies

Before adding new dependencies:

1. Check if really needed
2. Verify license compatibility
3. Consider package size
4. Update `requirements.txt`
5. Document in PR description

## Security

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. Email: security@your-domain.com
3. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Review Process

1. **Automated checks** run on PR
   - Syntax check
   - Import validation
   - Basic tests

2. **Manual review** by maintainers
   - Code quality
   - Test coverage
   - Documentation
   - Security considerations

3. **Feedback** provided within 48 hours
   - Address comments
   - Update PR

4. **Merge** when approved
   - Squash commits if needed
   - Update changelog

## Getting Help

- **Questions**: Open a discussion
- **Bugs**: Open an issue
- **Chat**: Join our Telegram group
- **Email**: support@your-domain.com

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project README

Thank you for contributing! 🎉
