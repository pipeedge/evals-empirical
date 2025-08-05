# MLR System - Multivocal Literature Review for AI/LLM Evaluation

A comprehensive automated research system for conducting Multivocal Literature Reviews (MLR) focused on AI/LLM evaluation frameworks. This system systematically gathers and analyzes practitioner-generated content from various sources to provide insights into current challenges, requirements, and best practices.

## 🎯 Research Questions Addressed

1. **RQ1**: What are the key challenges practitioners face when designing, implementing, and using AI/LLM evaluation frameworks?
2. **RQ2**: What are the key requirements and desired features practitioners articulate for next-generation AI/LLM evaluation frameworks and tools?
3. **RQ3**: How can these challenges and requirements be synthesized into a conceptual workflow to guide the integration of evaluation into the AI/LLM development lifecycle?

## 🏗️ System Architecture

The MLR system follows a three-phase pipeline:

### Phase 1: Data Acquisition
- **GitHub Collector**: Searches repositories, issues, and code files
- **Stack Overflow Collector**: Gathers questions and answers with relevant tags
- **Web Scraper**: Extracts articles from blogs and technical publications

### Phase 2: Filtering & Preprocessing
- **Content Filtering**: Removes irrelevant and low-quality content
- **Text Cleaning**: Normalizes and cleans text data
- **Duplicate Removal**: Eliminates redundant content
- **Language Detection**: Filters for English content

### Phase 3: Thematic Analysis
- **Frequency Analysis**: Identifies most common terms and phrases
- **Topic Modeling**: Uses LDA for automated topic discovery
- **LLM Analysis**: Leverages local LLM for advanced thematic extraction
- **Requirements Extraction**: Identifies functional and non-functional requirements
- **Workflow Synthesis**: Creates conceptual frameworks

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Local LLM** running via Ollama (as configured in your `Local_LLM_Samples.py`)
3. **Optional**: GitHub API token for enhanced rate limits

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Configuration

Edit `config.py` to customize:
- LLM endpoint and model settings
- Search keywords and research questions
- Data source configurations
- Filtering criteria
- Output directories

### Basic Usage

#### Run Complete Pipeline
```bash
python mlr_system.py --mode full
```

#### Run Individual Phases
```bash
# Data acquisition only
python mlr_system.py --mode acquisition

# Preprocessing only
python mlr_system.py --mode preprocessing

# Analysis only
python mlr_system.py --mode analysis
```

#### With GitHub Token (Recommended)
```bash
python mlr_system.py --mode full --github-token YOUR_GITHUB_TOKEN
```

#### Verbose Output
```bash
python mlr_system.py --mode full --verbose
```

## 📁 Output Structure

The system creates the following directory structure:

```
mlr/
├── data/
│   ├── raw/                    # Raw collected data
│   └── processed/              # Cleaned and filtered data
├── results/                    # Analysis results and reports
├── logs/                       # System logs
└── modules/                    # Core system modules
```

### Output Files

- **Raw Data**: JSON files with collected data from each source
- **Processed Data**: Cleaned documents in JSON and CSV formats
- **Analysis Results**: Comprehensive thematic analysis in JSON format
- **Reports**: Markdown reports with key findings and insights
- **Logs**: Detailed execution logs for debugging

## 🔧 Advanced Configuration

### Custom Data Sources

Add new domains to web scraping in `config.py`:
```python
WEB_SCRAPING_CONFIG = {
    "target_domains": [
        "your-domain.com",
        # ... existing domains
    ]
}
```

### Modify Search Terms

Update keywords in `config.py`:
```python
KEYWORDS = [
    "your custom keywords",
    # ... existing keywords
]
```

### LLM Configuration

Adjust LLM settings in `config.py`:
```python
LLM_CONFIG = {
    "base_url": "http://your-llm-endpoint:port",
    "model": "your-model-name",
    "timeout": 100
}
```

## 📊 Analysis Features

### Automated Theme Extraction
- Uses local LLM to identify key themes in practitioner discussions
- Categorizes challenges into technical, integration, and user experience domains
- Extracts requirements with priority levels

### Topic Modeling
- Latent Dirichlet Allocation for discovering hidden topics
- Automatic document-topic assignment
- Visualization of topic distributions

### Frequency Analysis
- Most common terms and phrases
- Bi-gram analysis for contextual understanding
- Technical term frequency tracking

### Workflow Synthesis
- LLM-generated conceptual workflows
- Integration of identified challenges and requirements
- Actionable recommendations for evaluation frameworks

## 🔍 Data Sources Covered

### GitHub
- Repositories related to LLM evaluation (⭐ 10+)
- Issues and discussions in relevant repositories
- Code files with evaluation implementations

### Stack Overflow
- Questions tagged with: `llm`, `evaluation`, `machine-learning`, `ai`, `nlp`, `rag`
- Accepted answers and high-scoring responses
- Discussion threads about evaluation challenges

### Technical Blogs & Publications
- Towards Data Science articles
- Medium publications
- Company engineering blogs (OpenAI, Google AI, etc.)
- Academic institution blogs

## 📈 Quality Assurance

### Content Filtering
- Minimum word count requirements (100-10,000 words)
- Exclusion of promotional content
- Language detection and filtering
- Technical relevance scoring

### Duplicate Detection
- Jaccard similarity for content comparison
- Cross-source deduplication
- Quality-based document selection

### Data Validation
- Metadata verification
- Source URL validation
- Content quality scoring

## 🛠️ Troubleshooting

### Common Issues

1. **LLM Connection Error**
   - Verify local LLM is running
   - Check endpoint URL in `config.py`
   - Ensure sufficient timeout values

2. **Rate Limiting**
   - Use GitHub API token
   - Increase delays in web scraping
   - Run acquisition in smaller batches

3. **Memory Issues**
   - Reduce batch sizes in configuration
   - Limit document count for analysis
   - Use analysis-only mode for large datasets

4. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')
   ```

### Logs and Debugging

- Check logs in `logs/` directory
- Use `--verbose` flag for detailed output
- Monitor individual phase execution

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built following MLR methodology best practices
- Integrates with Ollama for local LLM processing
- Leverages scikit-learn for machine learning components
- Uses NLTK for natural language processing

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review logs in the `logs/` directory
3. Open an issue with detailed error information

---

**Note**: This system is designed for research purposes. Ensure compliance with API terms of service and website scraping policies when collecting data from external sources. 