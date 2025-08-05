"""
Data Acquisition Module for MLR System
Handles automated collection from GitHub, Stack Overflow, and web sources
"""

import requests
import time
import logging
import json
import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd

from config import (
    GITHUB_CONFIG, STACKOVERFLOW_CONFIG, WEB_SCRAPING_CONFIG,
    HUGGINGFACE_CONFIG, KEYWORDS, OUTPUT_CONFIG
)

class GitHubCollector:
    """Collects data from GitHub repositories, issues, and discussions"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MLR-Research-Bot"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
            logging.info("GitHub token provided - higher rate limits available")
        else:
            logging.info("No GitHub token - using anonymous access with stricter rate limits")
        
    def search_repositories(self, query: str, min_stars: int = 10) -> List[Dict]:
        """Search for repositories related to the query"""
        url = f"{self.base_url}/search/repositories"
        params = {
            "q": f"{query} stars:>={min_stars}",
            "sort": "stars",
            "order": "desc",
            "per_page": 100
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            repositories = []
            for repo in data.get("items", []):
                repositories.append({
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo.get("description", ""),
                    "stars": repo["stargazers_count"],
                    "language": repo.get("language", ""),
                    "url": repo["html_url"],
                    "created_at": repo["created_at"],
                    "updated_at": repo["updated_at"],
                    "topics": repo.get("topics", [])
                })
            
            return repositories
            
        except Exception as e:
            logging.error(f"Error searching repositories: {e}")
            return []
    
    def get_repository_issues(self, repo_full_name: str, state: str = "all") -> List[Dict]:
        """Get issues from a specific repository with rate limiting"""
        url = f"{self.base_url}/repos/{repo_full_name}/issues"
        params = {
            "state": state,
            "per_page": 30,  # Reduced from 100
            "sort": "created",
            "direction": "desc"
        }
        
        try:
            # Add delay before issue requests
            time.sleep(1)
            
            response = requests.get(url, headers=self.headers, params=params)
            
            # Handle rate limiting gracefully
            if response.status_code == 403:
                logging.warning(f"GitHub rate limited for {repo_full_name} issues. Skipping.")
                return []
            
            response.raise_for_status()
            issues = response.json()
            
            processed_issues = []
            for issue in issues:
                # Skip pull requests
                if "pull_request" in issue:
                    continue
                    
                processed_issues.append({
                    "id": issue["id"],
                    "number": issue["number"],
                    "title": issue["title"],
                    "body": issue.get("body", ""),
                    "state": issue["state"],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "labels": [label["name"] for label in issue.get("labels", [])],
                    "comments": issue["comments"],
                    "url": issue["html_url"],
                    "repository": repo_full_name
                })
            
            return processed_issues
            
        except requests.exceptions.HTTPError as e:
            if "rate limit" in str(e).lower() or "403" in str(e):
                logging.warning(f"GitHub rate limited for {repo_full_name} issues. Skipping.")
                return []
            else:
                logging.error(f"Error getting issues for {repo_full_name}: {e}")
                return []
        except Exception as e:
            logging.error(f"Error getting issues for {repo_full_name}: {e}")
            return []
    
    def search_code(self, query: str, file_extension: str = None) -> List[Dict]:
        """Search for code files containing the query with rate limiting"""
        url = f"{self.base_url}/search/code"
        search_query = query
        if file_extension:
            search_query += f" extension:{file_extension.replace('.', '')}"
        
        params = {
            "q": search_query,
            "per_page": 30  # Reduced from 100 to be more conservative
        }
        
        try:
            # Add delay before code search requests (stricter rate limits)
            delay = GITHUB_CONFIG.get("code_search_delay", 3)
            time.sleep(delay)
            
            response = requests.get(url, headers=self.headers, params=params)
            
            # Handle rate limiting gracefully
            if response.status_code == 403:
                logging.warning(f"GitHub code search rate limited for query: {query}. Skipping code search.")
                return []
            
            response.raise_for_status()
            data = response.json()
            
            code_files = []
            for item in data.get("items", []):
                code_files.append({
                    "name": item["name"],
                    "path": item["path"],
                    "repository": item["repository"]["full_name"],
                    "url": item["html_url"],
                    "sha": item["sha"],
                    "score": item.get("score", 0)
                })
            
            return code_files
            
        except requests.exceptions.HTTPError as e:
            if "rate limit" in str(e).lower() or "403" in str(e):
                logging.warning(f"GitHub code search rate limited for query: {query}. Skipping.")
                return []
            else:
                logging.error(f"Error searching code: {e}")
                return []
        except Exception as e:
            logging.error(f"Error searching code: {e}")
            return []

class StackOverflowCollector:
    """Collects data from Stack Overflow using their API"""
    
    def __init__(self):
        self.base_url = "https://api.stackexchange.com/2.3"
        self.site = "stackoverflow"
    
    def search_questions(self, tags: List[str], min_score: int = 1) -> List[Dict]:
        """Search for questions with specific tags"""
        all_questions = []
        
        # Search for each tag individually to get more results
        for tag in tags[:3]:  # Limit to first 3 tags to avoid too many requests
            url = f"{self.base_url}/questions"
            params = {
                "site": self.site,
                "tagged": tag,  # Search individual tags
                "sort": "votes",
                "order": "desc",
                "pagesize": 50,  # Smaller page size per tag
                "filter": "withbody"
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for question in data.get("items", []):
                    if question.get("score", 0) >= min_score:
                        # Check if we already have this question
                        if not any(q["question_id"] == question["question_id"] for q in all_questions):
                            all_questions.append({
                                "question_id": question["question_id"],
                                "title": question["title"],
                                "body": question.get("body", ""),
                                "score": question["score"],
                                "view_count": question.get("view_count", 0),
                                "answer_count": question.get("answer_count", 0),
                                "tags": question.get("tags", []),
                                "creation_date": question["creation_date"],
                                "last_activity_date": question.get("last_activity_date"),
                                "url": question["link"],
                                "is_answered": question.get("is_answered", False),
                                "accepted_answer_id": question.get("accepted_answer_id")
                            })
                
                # Add delay between requests to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Error searching Stack Overflow questions for tag {tag}: {e}")
                continue
        
        return all_questions
    
    def get_answers(self, question_id: int) -> List[Dict]:
        """Get answers for a specific question"""
        url = f"{self.base_url}/questions/{question_id}/answers"
        params = {
            "site": self.site,
            "sort": "votes",
            "order": "desc",
            "filter": "withbody"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            answers = []
            for answer in data.get("items", []):
                answers.append({
                    "answer_id": answer["answer_id"],
                    "question_id": question_id,
                    "body": answer.get("body", ""),
                    "score": answer["score"],
                    "is_accepted": answer.get("is_accepted", False),
                    "creation_date": answer["creation_date"],
                    "last_activity_date": answer.get("last_activity_date")
                })
            
            return answers
            
        except Exception as e:
            logging.error(f"Error getting answers for question {question_id}: {e}")
            return []

class WebScraper:
    """Scrapes relevant articles from blogs and websites"""
    
    def __init__(self, request_delay: float = 1.0):
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
    
    def scrape_domain(self, domain: str, keywords: List[str], max_articles: int = 50) -> List[Dict]:
        """Scrape articles from a specific domain"""
        articles = []
        
        # This is a simplified implementation - in practice, you'd want domain-specific scrapers
        search_urls = self._generate_search_urls(domain, keywords)
        
        for search_url in search_urls[:max_articles]:
            try:
                time.sleep(self.request_delay)
                response = self.session.get(search_url, timeout=10)
                
                # Handle different HTTP errors gracefully
                if response.status_code == 404:
                    logging.warning(f"Page not found (404): {search_url}")
                    continue
                elif response.status_code == 500:
                    logging.warning(f"Server error (500): {search_url}")
                    continue
                elif response.status_code == 403:
                    logging.warning(f"Access denied (403): {search_url}")
                    continue
                
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                article_data = self._extract_article_data(soup, search_url, domain)
                
                if article_data:
                    articles.append(article_data)
                    
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout scraping {search_url}")
                continue
            except requests.exceptions.ConnectionError:
                logging.warning(f"Connection error scraping {search_url}")
                continue
            except Exception as e:
                logging.warning(f"Error scraping {search_url}: {e}")
                continue
        
        return articles
    
    def _generate_search_urls(self, domain: str, keywords: List[str]) -> List[str]:
        """Generate domain-specific URLs for high-value technical content"""
        urls = []
        
        # Evaluation-focused keywords
        eval_keywords = ["evaluation", "benchmark", "testing", "metrics", "assessment"]
        
        # OpenAI removed due to access restrictions (403 errors)
        # if False:  # Disabled - OpenAI blocks automated access
        #     pass
        
        if "anthropic.com" in domain:
            # Anthropic research on evaluation and safety
            urls.extend([
                "https://www.anthropic.com/research",
                "https://www.anthropic.com/research?category=safety",
            ])
        
        elif "ai.meta.com" in domain:
            # Meta AI blog posts
            urls.extend([
                "https://ai.meta.com/blog/",
                "https://ai.meta.com/blog/?page=1",
            ])
        
        elif "microsoft.com/research" in domain:
            # Microsoft Research blog
            for keyword in eval_keywords[:3]:
                urls.append(f"https://www.microsoft.com/en-us/research/blog/?s={keyword}")
        
        elif "amazon.science" in domain:
            # Amazon Science blog
            urls.extend([
                "https://www.amazon.science/blog",
                "https://www.amazon.science/blog?q=evaluation",
                "https://www.amazon.science/blog?q=benchmark",
            ])
        
        elif "nvidia.com" in domain:
            # NVIDIA Developer blog
            urls.extend([
                "https://developer.nvidia.com/blog/",
                "https://developer.nvidia.com/blog/tag/deep-learning/",
            ])
        
        elif "wandb.ai" in domain:
            # Weights & Biases articles about evaluation
            if "articles" in domain:
                for keyword in ["evaluation", "experiment-tracking", "model-monitoring"]:
                    urls.append(f"https://wandb.ai/articles/tagged/{keyword}")
            elif "fully-connected" in domain:
                urls.extend([
                    "https://wandb.ai/fully-connected/",
                    "https://wandb.ai/fully-connected/gradient-dissent/",
                ])
        
        elif "neptune.ai" in domain:
            # Neptune.ai blog
            urls.extend([
                "https://neptune.ai/blog",
                "https://neptune.ai/blog/category/model-evaluation",
            ])
        
        elif "comet.com" in domain:
            # Comet ML blog (only main blog, subdirectories don't exist)
            urls.extend([
                "https://www.comet.com/site/blog/",
            ])
        
        elif "huggingface.co/blog" in domain:
            # Hugging Face blog (search endpoints don't work, just use main blog)
            urls.extend([
                "https://huggingface.co/blog",
            ])
        
        elif "distill.pub" in domain:
            # Distill publications
            urls.append("https://distill.pub/")
        
        elif "googleblog.com" in domain:
            # Google AI blog
            urls.extend([
                "https://ai.googleblog.com/",
                "https://research.google/blog/",
            ])
        
        elif "deepmind.com" in domain:
            # DeepMind blog
            urls.extend([
                "https://deepmind.google/discover/blog/",
                "https://deepmind.google/research/",
            ])
        
        elif "paperswithcode.com" in domain:
            # Papers with Code blog
            urls.extend([
                "https://paperswithcode.com/blog",
                "https://paperswithcode.com/sota",
            ])
        
        return urls
    
    def _extract_article_data(self, soup: BeautifulSoup, url: str, domain: str) -> Optional[Dict]:
        """Extract article data with domain-specific optimization"""
        try:
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Domain-specific content extraction
            content_text = ""
            
            if "openai.com" in domain:
                # OpenAI blog structure
                content_selectors = ['article', '.post-content', '[data-testid="article-content"]', 'main']
            elif "anthropic.com" in domain:
                # Anthropic research structure
                content_selectors = ['.research-content', 'article', '.post-content', 'main']
            elif "ai.meta.com" in domain:
                # Meta AI blog structure
                content_selectors = ['article', '.blog-post-content', '.post-content', 'main']
            elif "microsoft.com" in domain:
                # Microsoft Research blog
                content_selectors = ['.blog-content', 'article', '.entry-content', 'main']
            elif "amazon.science" in domain:
                # Amazon Science blog
                content_selectors = ['.blog-post-content', 'article', '.post-content', 'main']
            elif "nvidia.com" in domain:
                # NVIDIA blog structure
                content_selectors = ['.blog-content', 'article', '.entry-content', 'main']
            elif "wandb.ai" in domain:
                # Weights & Biases article structure
                content_selectors = ['.article-content', 'article', '.post-content', 'main']
            elif "neptune.ai" in domain:
                # Neptune.ai blog structure
                content_selectors = ['.blog-post-content', 'article', '.entry-content', 'main']
            elif "huggingface.co" in domain:
                # Hugging Face blog structure
                content_selectors = ['.blog-content', 'article', '.prose', 'main']
            elif "distill.pub" in domain:
                # Distill publication structure
                content_selectors = ['d-article', 'article', '.post-content', 'main']
            elif "googleblog.com" in domain or "research.google" in domain:
                # Google AI blog structure
                content_selectors = ['.post-content', 'article', '.entry-content', 'main']
            elif "deepmind" in domain:
                # DeepMind blog structure
                content_selectors = ['.blog-content', 'article', '.post-content', 'main']
            elif "paperswithcode.com" in domain:
                # Papers with Code structure
                content_selectors = ['.blog-post-content', 'article', '.post-content', 'main']
            else:
                # Generic selectors
                content_selectors = [
                    'article', '.post-content', '.entry-content', 
                    '.content', 'main', '.blog-content'
                ]
            
            # Try to extract content using domain-specific selectors
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    content_text = content.get_text().strip()
                    if len(content_text) > 200:  # Ensure we got substantial content
                        break
            
            # Fallback extraction if domain-specific didn't work
            if not content_text or len(content_text) < 200:
                # Try more generic selectors
                fallback_selectors = ['article', 'main', '.content', 'body']
                for selector in fallback_selectors:
                    content = soup.select_one(selector)
                    if content:
                        content_text = content.get_text().strip()
                        if len(content_text) > 200:
                            break
            
            # Clean up content
            if content_text:
                # Remove excessive whitespace
                content_text = re.sub(r'\s+', ' ', content_text)
                # Remove common navigation text
                content_text = re.sub(r'(Skip to|Navigation|Menu|Footer|Subscribe|Share this|Related posts).*?(?=\.|$)', '', content_text, flags=re.IGNORECASE)
            
            if len(content_text) < 100:  # Skip if too short
                return None
            
            # Filter for evaluation-related content
            eval_indicators = [
                'evaluation', 'benchmark', 'testing', 'metric', 'assessment',
                'model performance', 'validation', 'llm', 'language model',
                'rag', 'retrieval', 'generation', 'ai safety'
            ]
            
            if not any(indicator in content_text.lower() for indicator in eval_indicators):
                return None  # Skip if not evaluation-related
            
            return {
                "title": title_text,
                "content": content_text[:5000],  # Limit content length
                "url": url,
                "domain": domain,
                "scraped_at": datetime.now().isoformat(),
                "word_count": len(content_text.split()),
                "relevance_score": sum(1 for indicator in eval_indicators if indicator in content_text.lower())
            }
            
        except Exception as e:
            logging.error(f"Error extracting article data from {url}: {e}")
            return None

class HuggingFaceCollector:
    """Collects data from Hugging Face Hub, papers, and blog"""
    
    def __init__(self, request_delay: float = 1.0):
        self.request_delay = request_delay
        self.hub_api_url = HUGGINGFACE_CONFIG["hub_api_url"]
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MLR-Research-Bot"
        })
    
    def search_models(self) -> List[Dict]:
        """Search for models with evaluation-related content"""
        models = []
        url = f"{self.hub_api_url}/models"
        
        try:
            for keyword in HUGGINGFACE_CONFIG["search_keywords"][:3]:
                params = {
                    "search": keyword,
                    "filter": "text-generation",
                    "sort": "downloads",
                    "direction": -1,
                    "limit": 20
                }
                
                time.sleep(self.request_delay)
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                for model in response.json():
                    # Get model details including card
                    model_info = self._get_model_details(model["id"])
                    if model_info:
                        models.append(model_info)
                
                if len(models) >= HUGGINGFACE_CONFIG["max_items_per_type"]:
                    break
                    
        except Exception as e:
            logging.error(f"Error searching Hugging Face models: {e}")
        
        return models
    
    def search_datasets(self) -> List[Dict]:
        """Search for evaluation datasets"""
        datasets = []
        url = f"{self.hub_api_url}/datasets"
        
        try:
            for keyword in HUGGINGFACE_CONFIG["search_keywords"][:3]:
                params = {
                    "search": keyword,
                    "filter": "task_categories:question-answering",
                    "sort": "downloads",
                    "direction": -1,
                    "limit": 20
                }
                
                time.sleep(self.request_delay)
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                for dataset in response.json():
                    # Get dataset details including card
                    dataset_info = self._get_dataset_details(dataset["id"])
                    if dataset_info:
                        datasets.append(dataset_info)
                
                if len(datasets) >= HUGGINGFACE_CONFIG["max_items_per_type"]:
                    break
                    
        except Exception as e:
            logging.error(f"Error searching Hugging Face datasets: {e}")
        
        return datasets
    
    def search_papers(self) -> List[Dict]:
        """Search for papers with evaluation content"""
        papers = []
        
        try:
            # Hugging Face papers search
            for keyword in HUGGINGFACE_CONFIG["search_keywords"][:3]:
                url = f"https://huggingface.co/papers"
                params = {"search": keyword}
                
                time.sleep(self.request_delay)
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                # Parse the HTML to extract paper information
                soup = BeautifulSoup(response.content, 'html.parser')
                paper_links = soup.find_all('a', href=lambda x: x and '/papers/' in x)
                
                for link in paper_links[:10]:
                    paper_url = f"https://huggingface.co{link['href']}"
                    paper_info = self._get_paper_details(paper_url)
                    if paper_info:
                        papers.append(paper_info)
                
                if len(papers) >= HUGGINGFACE_CONFIG["max_items_per_type"]:
                    break
                    
        except Exception as e:
            logging.error(f"Error searching Hugging Face papers: {e}")
        
        return papers
    
    def scrape_blog_posts(self) -> List[Dict]:
        """Scrape blog posts about evaluation"""
        blog_posts = []
        
        try:
            for keyword in HUGGINGFACE_CONFIG["search_keywords"][:3]:
                # Search blog posts
                search_url = f"https://huggingface.co/blog"
                
                time.sleep(self.request_delay)
                response = self.session.get(search_url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find blog post links
                blog_links = soup.find_all('a', href=lambda x: x and '/blog/' in x)
                
                for link in blog_links[:5]:
                    if any(kw.lower() in link.get_text().lower() for kw in ['evaluation', 'benchmark', 'metric']):
                        blog_url = f"https://huggingface.co{link['href']}"
                        blog_info = self._get_blog_post_details(blog_url)
                        if blog_info:
                            blog_posts.append(blog_info)
                
                if len(blog_posts) >= 20:
                    break
                    
        except Exception as e:
            logging.error(f"Error scraping Hugging Face blog: {e}")
        
        return blog_posts
    
    def _get_model_details(self, model_id: str) -> Optional[Dict]:
        """Get detailed model information including card content"""
        try:
            # Get model info
            url = f"{self.hub_api_url}/models/{model_id}"
            response = self.session.get(url)
            response.raise_for_status()
            model_data = response.json()
            
            # Get model card content
            card_content = ""
            try:
                card_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
                card_response = self.session.get(card_url)
                if card_response.status_code == 200:
                    card_content = card_response.text
            except:
                pass
            
            return {
                "id": model_id,
                "type": "model",
                "title": model_data.get("id", ""),
                "description": model_data.get("description", ""),
                "card_content": card_content,
                "tags": model_data.get("tags", []),
                "downloads": model_data.get("downloads", 0),
                "likes": model_data.get("likes", 0),
                "url": f"https://huggingface.co/{model_id}",
                "created_at": model_data.get("createdAt", ""),
                "pipeline_tag": model_data.get("pipeline_tag", "")
            }
            
        except Exception as e:
            logging.error(f"Error getting model details for {model_id}: {e}")
            return None
    
    def _get_dataset_details(self, dataset_id: str) -> Optional[Dict]:
        """Get detailed dataset information including card content"""
        try:
            # Get dataset info
            url = f"{self.hub_api_url}/datasets/{dataset_id}"
            response = self.session.get(url)
            response.raise_for_status()
            dataset_data = response.json()
            
            # Get dataset card content
            card_content = ""
            try:
                card_url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
                card_response = self.session.get(card_url)
                if card_response.status_code == 200:
                    card_content = card_response.text
            except:
                pass
            
            return {
                "id": dataset_id,
                "type": "dataset",
                "title": dataset_data.get("id", ""),
                "description": dataset_data.get("description", ""),
                "card_content": card_content,
                "tags": dataset_data.get("tags", []),
                "downloads": dataset_data.get("downloads", 0),
                "likes": dataset_data.get("likes", 0),
                "url": f"https://huggingface.co/datasets/{dataset_id}",
                "created_at": dataset_data.get("createdAt", "")
            }
            
        except Exception as e:
            logging.error(f"Error getting dataset details for {dataset_id}: {e}")
            return None
    
    def _get_paper_details(self, paper_url: str) -> Optional[Dict]:
        """Get paper details from Hugging Face papers"""
        try:
            response = self.session.get(paper_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract paper information
            title_elem = soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Extract abstract or description
            abstract_elem = soup.find('div', class_='prose') or soup.find('p')
            abstract = abstract_elem.get_text().strip() if abstract_elem else ""
            
            return {
                "type": "paper",
                "title": title,
                "abstract": abstract[:1000],  # Limit length
                "url": paper_url,
                "source": "huggingface_papers"
            }
            
        except Exception as e:
            logging.error(f"Error getting paper details from {paper_url}: {e}")
            return None
    
    def _get_blog_post_details(self, blog_url: str) -> Optional[Dict]:
        """Get blog post details"""
        try:
            response = self.session.get(blog_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract blog post information
            title_elem = soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Extract content
            content_elem = soup.find('article') or soup.find('div', class_='prose')
            content = content_elem.get_text().strip() if content_elem else ""
            
            if len(content) < 100:  # Skip if too short
                return None
            
            return {
                "type": "blog_post",
                "title": title,
                "content": content[:3000],  # Limit length
                "url": blog_url,
                "source": "huggingface_blog",
                "word_count": len(content.split())
            }
            
        except Exception as e:
            logging.error(f"Error getting blog post details from {blog_url}: {e}")
            return None

class DataAcquisitionOrchestrator:
    """Orchestrates the entire data acquisition process"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_collector = GitHubCollector(github_token)
        self.stackoverflow_collector = StackOverflowCollector()
        self.web_scraper = WebScraper(WEB_SCRAPING_CONFIG["request_delay"])
        self.huggingface_collector = HuggingFaceCollector(WEB_SCRAPING_CONFIG["request_delay"])
        
        # Create output directories
        os.makedirs(OUTPUT_CONFIG["raw_data_dir"], exist_ok=True)
    
    def collect_all_data(self) -> Dict[str, Any]:
        """Collect data from all sources"""
        logging.info("Starting data acquisition process")
        
        results = {
            "github": {
                "repositories": [],
                "issues": [],
                "code_files": []
            },
            "stackoverflow": {
                "questions": [],
                "answers": []
            },
            "web_articles": [],
            "huggingface": {
                "models": [],
                "datasets": [],
                "papers": [],
                "blog_posts": []
            }
        }
        
        # Collect GitHub data
        logging.info("Collecting GitHub data")
        for search_term in GITHUB_CONFIG["search_terms"]:
            repos = self.github_collector.search_repositories(
                search_term, GITHUB_CONFIG["min_stars"]
            )
            results["github"]["repositories"].extend(repos)
            
            # Get issues for each repository (limit to top 5 to avoid rate limits)
            for repo in repos[:5]:
                logging.info(f"Collecting issues from {repo['full_name']}...")
                issues = self.github_collector.get_repository_issues(repo["full_name"])
                results["github"]["issues"].extend(issues)
                
                # Stop if we have enough issues
                if len(results["github"]["issues"]) >= 100:
                    logging.info("Reached 100 issues limit, stopping issue collection")
                    break
            
            # Search for code files (only if enabled and with rate limiting)
            if GITHUB_CONFIG.get("enable_code_search", False):
                logging.info("Code search enabled - collecting code files with rate limiting")
                priority_extensions = [".py", ".md"]  # Focus on most relevant extensions
                for ext in priority_extensions:
                    if ext in GITHUB_CONFIG["file_extensions"]:
                        code_files = self.github_collector.search_code(search_term, ext)
                        results["github"]["code_files"].extend(code_files)
                        
                        # Stop if we have enough code files
                        if len(results["github"]["code_files"]) >= GITHUB_CONFIG.get("max_code_files", 50):
                            break
            else:
                logging.info("Code search disabled (to avoid rate limits) - skipping code file collection")
        
        # Collect Stack Overflow data
        logging.info("Collecting Stack Overflow data")
        questions = self.stackoverflow_collector.search_questions(
            STACKOVERFLOW_CONFIG["tags"], STACKOVERFLOW_CONFIG["min_score"]
        )
        results["stackoverflow"]["questions"] = questions
        
        # Get answers for questions (limit to avoid rate limits)
        for question in questions[:20]:
            answers = self.stackoverflow_collector.get_answers(question["question_id"])
            results["stackoverflow"]["answers"].extend(answers)
        
        # Collect web articles
        logging.info("Collecting web articles")
        for domain in WEB_SCRAPING_CONFIG["target_domains"]:
            articles = self.web_scraper.scrape_domain(
                domain, KEYWORDS, WEB_SCRAPING_CONFIG["max_articles_per_domain"]
            )
            results["web_articles"].extend(articles)
        
        # Collect Hugging Face data
        logging.info("Collecting Hugging Face data")
        if HUGGINGFACE_CONFIG["include_model_cards"]:
            models = self.huggingface_collector.search_models()
            results["huggingface"]["models"] = models
        
        if HUGGINGFACE_CONFIG["include_dataset_cards"]:
            datasets = self.huggingface_collector.search_datasets()
            results["huggingface"]["datasets"] = datasets
        
        if HUGGINGFACE_CONFIG["include_papers"]:
            papers = self.huggingface_collector.search_papers()
            results["huggingface"]["papers"] = papers
        
        if HUGGINGFACE_CONFIG["include_blog_posts"]:
            blog_posts = self.huggingface_collector.scrape_blog_posts()
            results["huggingface"]["blog_posts"] = blog_posts
        
        # Save raw data
        self._save_raw_data(results)
        
        logging.info(f"Data acquisition complete. Collected:")
        logging.info(f"  - {len(results['github']['repositories'])} GitHub repositories")
        logging.info(f"  - {len(results['github']['issues'])} GitHub issues")
        logging.info(f"  - {len(results['github']['code_files'])} GitHub code files")
        logging.info(f"  - {len(results['stackoverflow']['questions'])} Stack Overflow questions")
        logging.info(f"  - {len(results['stackoverflow']['answers'])} Stack Overflow answers")
        logging.info(f"  - {len(results['web_articles'])} web articles")
        logging.info(f"  - {len(results['huggingface']['models'])} Hugging Face models")
        logging.info(f"  - {len(results['huggingface']['datasets'])} Hugging Face datasets")
        logging.info(f"  - {len(results['huggingface']['papers'])} Hugging Face papers")
        logging.info(f"  - {len(results['huggingface']['blog_posts'])} Hugging Face blog posts")
        
        return results
    
    def _save_raw_data(self, data: Dict[str, Any]):
        """Save raw data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save GitHub data
        github_file = os.path.join(
            OUTPUT_CONFIG["raw_data_dir"], f"github_data_{timestamp}.json"
        )
        with open(github_file, 'w', encoding='utf-8') as f:
            json.dump(data["github"], f, indent=2, ensure_ascii=False)
        
        # Save Stack Overflow data
        so_file = os.path.join(
            OUTPUT_CONFIG["raw_data_dir"], f"stackoverflow_data_{timestamp}.json"
        )
        with open(so_file, 'w', encoding='utf-8') as f:
            json.dump(data["stackoverflow"], f, indent=2, ensure_ascii=False)
        
        # Save web articles
        web_file = os.path.join(
            OUTPUT_CONFIG["raw_data_dir"], f"web_articles_{timestamp}.json"
        )
        with open(web_file, 'w', encoding='utf-8') as f:
            json.dump(data["web_articles"], f, indent=2, ensure_ascii=False)
        
        # Save Hugging Face data
        hf_file = os.path.join(
            OUTPUT_CONFIG["raw_data_dir"], f"huggingface_data_{timestamp}.json"
        )
        with open(hf_file, 'w', encoding='utf-8') as f:
            json.dump(data["huggingface"], f, indent=2, ensure_ascii=False)
        
        logging.info(f"Raw data saved to {OUTPUT_CONFIG['raw_data_dir']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    orchestrator = DataAcquisitionOrchestrator()
    data = orchestrator.collect_all_data() 