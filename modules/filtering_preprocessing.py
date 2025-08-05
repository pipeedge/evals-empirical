"""
Filtering and Preprocessing Module for MLR System
Handles data cleaning, filtering, and preprocessing
"""

import re
import json
import logging
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import html
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect
import string

from config import FILTERING_CONFIG, OUTPUT_CONFIG

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class TextCleaner:
    """Handles text cleaning and normalization"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.exclude_keywords = [kw.lower() for kw in FILTERING_CONFIG["exclude_keywords"]]
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities"""
        if not text:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text()
        
        # Decode HTML entities
        clean_text = html.unescape(clean_text)
        
        return clean_text
    
    def clean_text(self, text: str) -> str:
        """Comprehensive text cleaning"""
        if not text:
            return ""
        
        # Remove HTML
        text = self.clean_html(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation (but keep some for sentence structure)
        # Keep periods, commas, question marks, exclamation marks
        text = re.sub(r'[^\w\s\.\,\?\!]', ' ', text)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if not text:
            return []
        
        try:
            sentences = sent_tokenize(text)
            return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
        except:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [sent.strip() for sent in sentences if len(sent.strip()) > 10]

class ContentFilter:
    """Filters content based on inclusion/exclusion criteria"""
    
    def __init__(self):
        self.min_word_count = FILTERING_CONFIG["min_word_count"]
        self.max_word_count = FILTERING_CONFIG["max_word_count"]
        self.exclude_keywords = [kw.lower() for kw in FILTERING_CONFIG["exclude_keywords"]]
        self.supported_languages = FILTERING_CONFIG["languages"]
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the text"""
        try:
            if len(text.strip()) < 50:  # Too short for reliable detection
                return None
            return detect(text)
        except:
            return None
    
    def is_relevant_content(self, text: str, title: str = "") -> bool:
        """Check if content is relevant based on filtering criteria"""
        if not text:
            return False
        
        # Check word count
        word_count = len(text.split())
        if word_count < self.min_word_count or word_count > self.max_word_count:
            return False
        
        # Check for excluded keywords
        combined_text = (title + " " + text).lower()
        for keyword in self.exclude_keywords:
            if keyword in combined_text:
                return False
        
        # Check language
        language = self.detect_language(text)
        if language and language not in self.supported_languages:
            return False
        
        return True
    
    def has_technical_content(self, text: str) -> bool:
        """Check if content has technical relevance to AI/LLM evaluation"""
        technical_indicators = [
            'evaluation', 'benchmark', 'metric', 'testing', 'validation',
            'llm', 'language model', 'ai model', 'machine learning',
            'performance', 'accuracy', 'precision', 'recall', 'f1',
            'rag', 'retrieval', 'embedding', 'transformer', 'bert',
            'gpt', 'hugging face', 'pytorch', 'tensorflow', 'sklearn',
            'dataset', 'training', 'inference', 'fine-tuning'
        ]
        
        text_lower = text.lower()
        score = sum(1 for indicator in technical_indicators if indicator in text_lower)
        
        # Require at least 2 technical indicators
        return score >= 2

class GitHubDataProcessor:
    """Processes GitHub data (repositories, issues, code files)"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.content_filter = ContentFilter()
    
    def process_repositories(self, repositories: List[Dict]) -> List[Dict]:
        """Process GitHub repositories"""
        processed = []
        
        for repo in repositories:
            # Combine description and topics for content analysis
            content = f"{repo.get('description', '')} {' '.join(repo.get('topics', []))}"
            content = self.text_cleaner.clean_text(content)
            
            if self.content_filter.is_relevant_content(content, repo.get('name', '')):
                processed_repo = {
                    'source': 'github_repository',
                    'id': repo['full_name'],
                    'title': repo['name'],
                    'content': content,
                    'url': repo['url'],
                    'metadata': {
                        'stars': repo['stars'],
                        'language': repo.get('language', ''),
                        'created_at': repo['created_at'],
                        'topics': repo.get('topics', [])
                    }
                }
                processed.append(processed_repo)
        
        return processed
    
    def process_issues(self, issues: List[Dict]) -> List[Dict]:
        """Process GitHub issues"""
        processed = []
        
        for issue in issues:
            # Combine title and body
            content = f"{issue.get('title', '')} {issue.get('body', '')}"
            content = self.text_cleaner.clean_text(content)
            
            if (self.content_filter.is_relevant_content(content, issue.get('title', '')) and
                self.content_filter.has_technical_content(content)):
                
                processed_issue = {
                    'source': 'github_issue',
                    'id': f"{issue['repository']}#{issue['number']}",
                    'title': issue['title'],
                    'content': content,
                    'url': issue['url'],
                    'metadata': {
                        'repository': issue['repository'],
                        'state': issue['state'],
                        'labels': issue.get('labels', []),
                        'comments': issue['comments'],
                        'created_at': issue['created_at']
                    }
                }
                processed.append(processed_issue)
        
        return processed

class StackOverflowDataProcessor:
    """Processes Stack Overflow data"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.content_filter = ContentFilter()
    
    def process_questions(self, questions: List[Dict], answers: List[Dict]) -> List[Dict]:
        """Process Stack Overflow questions and their answers"""
        processed = []
        
        # Create a mapping of answers by question_id
        answers_by_question = {}
        for answer in answers:
            qid = answer['question_id']
            if qid not in answers_by_question:
                answers_by_question[qid] = []
            answers_by_question[qid].append(answer)
        
        for question in questions:
            # Combine question title and body
            question_content = f"{question.get('title', '')} {question.get('body', '')}"
            question_content = self.text_cleaner.clean_text(question_content)
            
            # Add top answers (max 3)
            qid = question['question_id']
            answer_content = ""
            if qid in answers_by_question:
                top_answers = sorted(answers_by_question[qid], 
                                   key=lambda x: x['score'], reverse=True)[:3]
                answer_texts = [self.text_cleaner.clean_text(ans.get('body', '')) 
                              for ans in top_answers]
                answer_content = " ".join(answer_texts)
            
            # Combine question and answers
            full_content = f"{question_content} {answer_content}"
            
            if (self.content_filter.is_relevant_content(full_content, question.get('title', '')) and
                self.content_filter.has_technical_content(full_content)):
                
                processed_question = {
                    'source': 'stackoverflow',
                    'id': str(question['question_id']),
                    'title': question['title'],
                    'content': full_content,
                    'url': question['url'],
                    'metadata': {
                        'score': question['score'],
                        'view_count': question.get('view_count', 0),
                        'answer_count': question.get('answer_count', 0),
                        'tags': question.get('tags', []),
                        'is_answered': question.get('is_answered', False),
                        'creation_date': question['creation_date']
                    }
                }
                processed.append(processed_question)
        
        return processed

class WebArticleProcessor:
    """Processes web articles from blogs and other sources"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.content_filter = ContentFilter()
    
    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process web articles"""
        processed = []
        
        for article in articles:
            content = self.text_cleaner.clean_text(article.get('content', ''))
            title = article.get('title', '')
            
            if (self.content_filter.is_relevant_content(content, title) and
                self.content_filter.has_technical_content(content)):
                
                processed_article = {
                    'source': 'web_article',
                    'id': article['url'],
                    'title': title,
                    'content': content,
                    'url': article['url'],
                    'metadata': {
                        'domain': article.get('domain', ''),
                        'word_count': article.get('word_count', 0),
                        'scraped_at': article.get('scraped_at', '')
                    }
                }
                processed.append(processed_article)
        
        return processed

class HuggingFaceDataProcessor:
    """Processes Hugging Face data (models, datasets, papers, blog posts)"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.content_filter = ContentFilter()
    
    def process_huggingface_data(self, hf_data: Dict) -> List[Dict]:
        """Process all Hugging Face data types"""
        processed = []
        
        # Process models
        for model in hf_data.get('models', []):
            content = f"{model.get('description', '')} {model.get('card_content', '')}"
            content = self.text_cleaner.clean_text(content)
            
            if (self.content_filter.is_relevant_content(content, model.get('title', '')) and
                self.content_filter.has_technical_content(content)):
                
                processed_model = {
                    'source': 'huggingface_model',
                    'id': model['id'],
                    'title': model['title'],
                    'content': content,
                    'url': model['url'],
                    'metadata': {
                        'type': 'model',
                        'downloads': model.get('downloads', 0),
                        'likes': model.get('likes', 0),
                        'tags': model.get('tags', []),
                        'pipeline_tag': model.get('pipeline_tag', ''),
                        'created_at': model.get('created_at', '')
                    }
                }
                processed.append(processed_model)
        
        # Process datasets
        for dataset in hf_data.get('datasets', []):
            content = f"{dataset.get('description', '')} {dataset.get('card_content', '')}"
            content = self.text_cleaner.clean_text(content)
            
            if (self.content_filter.is_relevant_content(content, dataset.get('title', '')) and
                self.content_filter.has_technical_content(content)):
                
                processed_dataset = {
                    'source': 'huggingface_dataset',
                    'id': dataset['id'],
                    'title': dataset['title'],
                    'content': content,
                    'url': dataset['url'],
                    'metadata': {
                        'type': 'dataset',
                        'downloads': dataset.get('downloads', 0),
                        'likes': dataset.get('likes', 0),
                        'tags': dataset.get('tags', []),
                        'created_at': dataset.get('created_at', '')
                    }
                }
                processed.append(processed_dataset)
        
        # Process papers
        for paper in hf_data.get('papers', []):
            content = paper.get('abstract', '')
            content = self.text_cleaner.clean_text(content)
            
            if (self.content_filter.is_relevant_content(content, paper.get('title', '')) and
                self.content_filter.has_technical_content(content)):
                
                processed_paper = {
                    'source': 'huggingface_paper',
                    'id': paper['url'],
                    'title': paper['title'],
                    'content': content,
                    'url': paper['url'],
                    'metadata': {
                        'type': 'paper',
                        'source': 'huggingface_papers'
                    }
                }
                processed.append(processed_paper)
        
        # Process blog posts
        for blog_post in hf_data.get('blog_posts', []):
            content = self.text_cleaner.clean_text(blog_post.get('content', ''))
            
            if (self.content_filter.is_relevant_content(content, blog_post.get('title', '')) and
                self.content_filter.has_technical_content(content)):
                
                processed_blog = {
                    'source': 'huggingface_blog',
                    'id': blog_post['url'],
                    'title': blog_post['title'],
                    'content': content,
                    'url': blog_post['url'],
                    'metadata': {
                        'type': 'blog_post',
                        'word_count': blog_post.get('word_count', 0),
                        'source': 'huggingface_blog'
                    }
                }
                processed.append(processed_blog)
        
        return processed

class DuplicateRemover:
    """Removes duplicate content"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def remove_duplicates(self, documents: List[Dict]) -> List[Dict]:
        """Remove duplicate documents"""
        unique_documents = []
        
        for doc in documents:
            is_duplicate = False
            doc_content = doc.get('content', '')
            
            for unique_doc in unique_documents:
                unique_content = unique_doc.get('content', '')
                similarity = self.calculate_similarity(doc_content, unique_content)
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    # Keep the document with higher quality (more metadata, longer content, etc.)
                    if len(doc_content) > len(unique_content):
                        # Replace the existing document
                        unique_documents.remove(unique_doc)
                        unique_documents.append(doc)
                    break
            
            if not is_duplicate:
                unique_documents.append(doc)
        
        return unique_documents

class FilteringPreprocessingOrchestrator:
    """Orchestrates the entire filtering and preprocessing pipeline"""
    
    def __init__(self):
        self.github_processor = GitHubDataProcessor()
        self.stackoverflow_processor = StackOverflowDataProcessor()
        self.web_processor = WebArticleProcessor()
        self.huggingface_processor = HuggingFaceDataProcessor()
        self.duplicate_remover = DuplicateRemover()
        
        # Create output directories
        os.makedirs(OUTPUT_CONFIG["processed_data_dir"], exist_ok=True)
    
    def process_all_data(self, raw_data_dir: str = None) -> List[Dict]:
        """Process all raw data into clean, filtered documents"""
        if raw_data_dir is None:
            raw_data_dir = OUTPUT_CONFIG["raw_data_dir"]
        
        logging.info("Starting data filtering and preprocessing")
        
        all_documents = []
        
        # Find the latest raw data files
        raw_files = os.listdir(raw_data_dir)
        
        # Process GitHub data
        github_files = [f for f in raw_files if f.startswith('github_data_')]
        if github_files:
            latest_github = sorted(github_files)[-1]
            github_path = os.path.join(raw_data_dir, latest_github)
            
            with open(github_path, 'r', encoding='utf-8') as f:
                github_data = json.load(f)
            
            # Process repositories
            repo_docs = self.github_processor.process_repositories(
                github_data.get('repositories', [])
            )
            all_documents.extend(repo_docs)
            
            # Process issues
            issue_docs = self.github_processor.process_issues(
                github_data.get('issues', [])
            )
            all_documents.extend(issue_docs)
            
            logging.info(f"Processed {len(repo_docs)} repositories and {len(issue_docs)} issues")
        
        # Process Stack Overflow data
        so_files = [f for f in raw_files if f.startswith('stackoverflow_data_')]
        if so_files:
            latest_so = sorted(so_files)[-1]
            so_path = os.path.join(raw_data_dir, latest_so)
            
            with open(so_path, 'r', encoding='utf-8') as f:
                so_data = json.load(f)
            
            so_docs = self.stackoverflow_processor.process_questions(
                so_data.get('questions', []), so_data.get('answers', [])
            )
            all_documents.extend(so_docs)
            
            logging.info(f"Processed {len(so_docs)} Stack Overflow discussions")
        
        # Process web articles
        web_files = [f for f in raw_files if f.startswith('web_articles_')]
        if web_files:
            latest_web = sorted(web_files)[-1]
            web_path = os.path.join(raw_data_dir, latest_web)
            
            with open(web_path, 'r', encoding='utf-8') as f:
                web_data = json.load(f)
            
            web_docs = self.web_processor.process_articles(web_data)
            all_documents.extend(web_docs)
            
            logging.info(f"Processed {len(web_docs)} web articles")
        
        # Process Hugging Face data
        hf_files = [f for f in raw_files if f.startswith('huggingface_data_')]
        if hf_files:
            latest_hf = sorted(hf_files)[-1]
            hf_path = os.path.join(raw_data_dir, latest_hf)
            
            with open(hf_path, 'r', encoding='utf-8') as f:
                hf_data = json.load(f)
            
            hf_docs = self.huggingface_processor.process_huggingface_data(hf_data)
            all_documents.extend(hf_docs)
            
            logging.info(f"Processed {len(hf_docs)} Hugging Face items")
        
        # Remove duplicates
        logging.info("Removing duplicates...")
        initial_count = len(all_documents)
        all_documents = self.duplicate_remover.remove_duplicates(all_documents)
        final_count = len(all_documents)
        
        logging.info(f"Removed {initial_count - final_count} duplicates. "
                    f"Final document count: {final_count}")
        
        # Save processed data
        self._save_processed_data(all_documents)
        
        return all_documents
    
    def _save_processed_data(self, documents: List[Dict]):
        """Save processed documents to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            OUTPUT_CONFIG["processed_data_dir"], 
            f"processed_documents_{timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        # Also save as CSV for easier analysis
        csv_file = os.path.join(
            OUTPUT_CONFIG["processed_data_dir"], 
            f"processed_documents_{timestamp}.csv"
        )
        
        df_data = []
        for doc in documents:
            row = {
                'source': doc['source'],
                'id': doc['id'],
                'title': doc['title'],
                'content_preview': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                'content_length': len(doc['content']),
                'url': doc['url']
            }
            # Add metadata fields
            for key, value in doc.get('metadata', {}).items():
                row[f'meta_{key}'] = value
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logging.info(f"Processed data saved to {output_file} and {csv_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    processor = FilteringPreprocessingOrchestrator()
    documents = processor.process_all_data() 