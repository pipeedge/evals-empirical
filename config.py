"""
Configuration file for the MLR (Multivocal Literature Review) system
"""

# LLM Configuration (based on Local_LLM_Samples.py)
LLM_CONFIG = {
    "base_url": "http://10.33.205.34:11112",
    "model": "llama3.1:70b",
    "timeout": 100
}

# Research Questions and Keywords
RESEARCH_QUESTIONS = [
    "What are the key challenges practitioners face when designing, implementing, and using AI/LLM evaluation frameworks?",
    "What are the key requirements and desired features practitioners articulate for next-generation AI/LLM evaluation frameworks and tools?",
    "How can these challenges and requirements be synthesized into a conceptual workflow to guide the integration of evaluation into the AI/LLM development lifecycle?"
]

# Keywords for automated search
KEYWORDS = [
    "LLM evaluation", "model testing", "RAG assessment", "agent benchmark",
    "AI evaluation framework", "language model testing", "LLM benchmark",
    "evaluation metrics", "model validation", "AI testing tools",
    "evaluation pipeline", "LLM performance", "model assessment",
    "evaluation automation", "testing framework", "AI model evaluation"
]

# Data Sources Configuration
GITHUB_CONFIG = {
    "search_terms": [
        "LLM evaluation", "language model testing", "AI evaluation framework",
        "model benchmark", "evaluation metrics", "RAG evaluation"
    ],
    "file_extensions": [".py", ".md", ".txt", ".json"],
    "min_stars": 10,
    "issues_only": False,
    "enable_code_search": False,  # Disabled by default due to strict rate limits
    "max_code_files": 50,  # Limit total code files to collect
    "code_search_delay": 3  # Seconds between code search requests
}

STACKOVERFLOW_CONFIG = {
    "tags": ["llm", "language-model", "machine-learning", "nlp", "chatgpt", "openai", "huggingface", "transformers"],  # Individual tags, not all required
    "min_score": 1,
    "min_answers": 1,
    "accepted_answer_only": False
}

WEB_SCRAPING_CONFIG = {
    "target_domains": [
        # Company engineering blogs (highest value for practitioner challenges)
        # Note: OpenAI blog removed due to access restrictions (403 errors)
        "www.anthropic.com/research",         # Anthropic research
        "ai.meta.com/blog",                   # Meta AI/LLaMA evaluation ✅ 200
        "www.microsoft.com/research/blog",    # Microsoft Research evaluation
        "www.amazon.science/blog",            # AWS ML/Bedrock evaluation
        "developer.nvidia.com/blog",          # NVIDIA Developer blog ✅ 200
        
        # MLOps platforms (evaluation workflows and challenges)
        "wandb.ai/articles",                  # Weights & Biases articles ✅ 200
        "wandb.ai/fully-connected",           # W&B Fully Connected ✅ 200
        "neptune.ai/blog",                    # ML experiment management
        "www.comet.com/site/blog",            # Comet ML blog ✅ 200
        
        # High-quality technical content
        "huggingface.co/blog",                # HF evaluation insights
        "distill.pub",                        # Technical explanations
        "ai.googleblog.com",                  # Google AI evaluation research
        "deepmind.google/discover/blog",      # DeepMind blog
        
        # Evaluation-focused platforms
        "paperswithcode.com/blog",            # Benchmark discussions
    ],
    "request_delay": 1.5,  # Increased delay for better scraping success
    "max_articles_per_domain": 30  # Reduced per domain, but more quality domains
}

# Hugging Face Configuration
HUGGINGFACE_CONFIG = {
    "hub_api_url": "https://huggingface.co/api",  # Note: Some endpoints may redirect
    "search_keywords": [
        "evaluation", "benchmark", "metric", "testing", "validation",
        "rag evaluation", "llm evaluation", "model assessment"
    ],
    "model_filters": {
        "pipeline_tag": ["text-generation", "question-answering", "text-classification"],
        "library": ["transformers", "sentence-transformers"],
        "min_downloads": 1000
    },
    "dataset_filters": {
        "task_categories": ["question-answering", "text-generation", "text-classification"],
        "min_downloads": 100
    },
    "max_items_per_type": 100,
    "include_model_cards": True,
    "include_dataset_cards": True,
    "include_papers": True,
    "include_discussions": True,
    "include_blog_posts": True
}

# Filtering and Preprocessing Configuration
FILTERING_CONFIG = {
    "min_word_count": 100,
    "max_word_count": 10000,
    "exclude_keywords": [
        "advertisement", "sponsored", "buy now", "sale", "discount",
        "subscribe", "newsletter", "commercial"
    ],
    "languages": ["en"]
}

# Thematic Analysis Configuration
ANALYSIS_CONFIG = {
    "topic_modeling": {
        "num_topics": 10,
        "passes": 10,
        "alpha": "auto",
        "eta": "auto"
    },
    "clustering": {
        "min_cluster_size": 5,
        "metric": "euclidean"
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    "data_dir": "data",
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "results_dir": "results",
    "logs_dir": "logs"
} 