# Contributing to VoiceAccess

Thank you for your interest in contributing to VoiceAccess! We welcome contributions from researchers, developers, linguists, and language communities worldwide.

## 🌟 Ways to Contribute

### 1. Adding Language Support
- **Dataset Contribution**: Share transcribed audio data for your language
- **Language Adapters**: Implement language-specific processing modules
- **Phoneme Mappings**: Provide phonetic transcriptions for your language
- **Testing**: Validate model performance on your language

### 2. Technical Contributions
- **Model Improvements**: Enhance existing architectures or add new ones
- **Feature Development**: Implement new features or improve existing ones
- **Bug Fixes**: Help us identify and fix issues
- **Performance Optimization**: Make the system faster and more efficient

### 3. Documentation
- **Translations**: Translate documentation to other languages
- **Tutorials**: Create guides for specific use cases
- **API Documentation**: Improve code documentation
- **Examples**: Add more usage examples

### 4. Community Support
- **Answer Questions**: Help others in discussions and issues
- **Review PRs**: Provide feedback on pull requests
- **Testing**: Test new features and report bugs
- **Advocacy**: Spread awareness about the project

## 🛠️ Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/VoiceAccess.git
   cd VoiceAccess
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

## 📝 Contribution Process

1. **Create an Issue**: Discuss your idea or bug report
2. **Fork & Branch**: Create a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit Changes**: Use clear, descriptive commit messages
4. **Write Tests**: Ensure your code is tested
5. **Run Checks**: 
   ```bash
   # Format code
   black src/ tests/
   
   # Check style
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest tests/
   ```
6. **Push Branch**: Push to your fork (`git push origin feature/amazing-feature`)
7. **Open PR**: Create a Pull Request with a clear description

## 📋 Pull Request Guidelines

### PR Title Format
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/modifications
- `chore:` Maintenance tasks

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed code
- [ ] Commented complex parts
- [ ] No new warnings
```

## 🌐 Adding a New Language

### Step 1: Prepare Your Data
```
data/
└── your_language/
    ├── train/
    │   ├── audio/
    │   └── transcripts.txt
    ├── dev/
    │   ├── audio/
    │   └── transcripts.txt
    └── test/
        ├── audio/
        └── transcripts.txt
```

### Step 2: Create Language Configuration
```yaml
# configs/languages/your_language.yaml
language_code: "xx"
language_name: "Your Language"
script: "Latin"  # or Cyrillic, Arabic, etc.
direction: "ltr"  # or rtl
phoneme_set: "ipa"  # or custom
sample_rate: 16000
```

### Step 3: Implement Language Adapter
```python
# src/languages/your_language.py
from ..languages.base_adapter import BaseLanguageAdapter

class YourLanguageAdapter(BaseLanguageAdapter):
    def __init__(self, config):
        super().__init__(config)
        # Language-specific initialization
    
    def preprocess_text(self, text):
        # Language-specific text processing
        return processed_text
    
    def postprocess_text(self, text):
        # Language-specific post-processing
        return final_text
```

### Step 4: Submit Your Contribution
1. Add documentation for your language
2. Include sample audio/text pairs
3. Provide evaluation results
4. Create a PR with clear description

## 🐛 Reporting Issues

### Bug Reports
Include:
- VoiceAccess version
- Python version
- OS information
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests
Include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Impact on existing features

## 💻 Code Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints where possible
- Write docstrings for all public functions
- Keep functions focused and small
- Use meaningful variable names

### Example Code Style
```python
def transcribe_audio(
    audio_path: Union[str, Path],
    language: Optional[str] = None,
    model: Optional[BaseASRModel] = None
) -> Dict[str, Any]:
    """
    Transcribe audio file to text.
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'en', 'es')
        model: Pre-loaded ASR model
        
    Returns:
        Dictionary containing transcription and metadata
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If language not supported
    """
    # Implementation
    pass
```

## 🧪 Testing Guidelines

- Write tests for new features
- Maintain test coverage above 80%
- Use pytest fixtures for common setups
- Test edge cases and error conditions

### Test Structure
```python
def test_feature_normal_case():
    """Test normal operation"""
    assert expected == actual

def test_feature_edge_case():
    """Test edge cases"""
    with pytest.raises(ValueError):
        problematic_operation()
```

## 📚 Documentation Standards

- Update README for significant changes
- Add docstrings to all public APIs
- Include usage examples
- Update configuration docs
- Add to changelog

## 🤝 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing private information
- Unethical or unprofessional conduct

## 📫 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat and support
- **Email**: voiceaccess@openimpactai.org

## 🎉 Recognition

Contributors will be:
- Listed in our [Contributors](CONTRIBUTORS.md) file
- Mentioned in release notes
- Invited to our contributor community
- Given credit in presentations/papers

Thank you for helping make speech recognition accessible to all languages! 🌍