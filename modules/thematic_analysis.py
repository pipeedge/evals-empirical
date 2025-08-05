"""
Thematic Analysis Module for MLR System
Uses NLP techniques and local LLM for semi-automated thematic analysis
"""

import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

# NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# LLM integration
from langchain_ollama import OllamaLLM

from config import LLM_CONFIG, ANALYSIS_CONFIG, OUTPUT_CONFIG, RESEARCH_QUESTIONS

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class LLMAnalyst:
    """Uses local LLM for advanced thematic analysis"""
    
    def __init__(self):
        self.llm = OllamaLLM(
            base_url=LLM_CONFIG["base_url"],
            model=LLM_CONFIG["model"],
            timeout=LLM_CONFIG["timeout"]
        )
    
    def extract_themes_from_text(self, text: str, context: str = "", source_info: Dict = None) -> Dict:
        """Extract themes from text using LLM with source attribution"""
        source_ref = f"{source_info.get('source', 'Unknown')}:{source_info.get('id', 'N/A')}" if source_info else "Unknown"
        
        prompt = f"""
        Analyze the following text and extract key themes related to AI/LLM evaluation frameworks.
        Focus on challenges, requirements, tools, and methodologies mentioned.
        
        Context: {context}
        Source: {source_ref}
        
        Text to analyze:
        {text[:3000]}  # Limit text length
        
        For each theme, provide:
        1. Theme name
        2. Brief description  
        3. Supporting evidence (direct quote from text)
        4. Confidence level (High/Medium/Low)
        
        Format:
        THEME: [Theme Name]
        DESCRIPTION: [Brief description]
        EVIDENCE: "[Direct quote supporting this theme]"
        CONFIDENCE: [High/Medium/Low]
        
        Themes:
        """
        
        try:
            response = self.llm.invoke(prompt)
            themes_with_evidence = self._parse_themes_with_evidence(response, source_info)
            return themes_with_evidence if themes_with_evidence else []
        except Exception as e:
            logging.error(f"Error extracting themes with LLM: {e}")
            # Return empty list to maintain consistency
            return []
    
    def categorize_challenges(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize challenges mentioned in documents using LLM"""
        challenge_categories = {
            "Technical Challenges": [],
            "Integration Challenges": [],
            "Data Quality Challenges": [],
            "Scalability Challenges": [],
            "User Experience Challenges": [],
            "Performance Challenges": []
        }
        
        for doc in documents:
            if len(doc['content']) < 100:  # Skip very short documents
                continue
                
            prompt = f"""
            Analyze the following text about AI/LLM evaluation and identify which types of challenges are mentioned.
            
            Categories:
            - Technical Challenges: Implementation, coding, algorithms
            - Integration Challenges: Connecting systems, workflow integration
            - Data Quality Challenges: Data issues, quality, availability
            - Scalability Challenges: Performance at scale, resource constraints
            - User Experience Challenges: Usability, interface, adoption
            - Performance Challenges: Speed, accuracy, efficiency
            
            Text: {doc['content'][:2000]}
            
            Respond with only the category names that apply (one per line), or "None" if no challenges are mentioned:
            """
            
            try:
                response = self.llm.invoke(prompt)
                categories = self._parse_categories_response(response)
                
                for category in categories:
                    if category in challenge_categories:
                        challenge_categories[category].append(doc)
                        
            except Exception as e:
                logging.error(f"Error categorizing challenges: {e}")
                continue
        
        return challenge_categories
    
    def extract_requirements(self, documents: List[Dict]) -> List[Dict]:
        """Extract requirements and desired features using LLM"""
        requirements = []
        
        for doc in documents:
            if len(doc['content']) < 100:
                continue
                
            prompt = f"""
            Analyze the following text and extract any requirements, desired features, or needs mentioned for AI/LLM evaluation frameworks.
            
            Text: {doc['content'][:2000]}
            
            Format your response as:
            REQUIREMENT: [Brief description of requirement]
            PRIORITY: [High/Medium/Low based on context]
            CATEGORY: [Functional/Non-functional/Technical/User]
            
            If no requirements are mentioned, respond with "None".
            
            Requirements:
            """
            
            try:
                response = self.llm.invoke(prompt)
                parsed_requirements = self._parse_requirements_response(response, doc)
                requirements.extend(parsed_requirements)
                
            except Exception as e:
                logging.error(f"Error extracting requirements: {e}")
                continue
        
        return requirements
    
    def synthesize_workflow(self, themes: List[str], challenges: Dict[str, List[Dict]], 
                           requirements: List[Dict]) -> str:
        """Synthesize findings into a comprehensive conceptual workflow with evidence"""
        
        # Enhanced themes with evidence if available
        if isinstance(themes, list) and themes and isinstance(themes[0], dict):
            themes_text = "\n".join([
                f"- {theme.get('theme', 'Unknown')}: {theme.get('description', 'No description')}"
                for theme in themes[:8] if theme.get('theme')
            ])
        else:
            # Handle legacy format or fallback
            themes_text = "\n".join([f"- {theme}" for theme in themes[:10] if theme])
        
        challenges_summary = "\n".join([
            f"- {category}: {len(docs)} mentions from practitioner reports" 
            for category, docs in challenges.items() if docs
        ])
        
        requirements_summary = "\n".join([
            f"- {req['requirement'][:100]} (Priority: {req.get('priority', 'N/A')}, Source: {req.get('source', 'Unknown')})" 
            for req in requirements[:12]
        ])
        
        prompt = f"""
        Based on comprehensive analysis of {sum(len(docs) for docs in challenges.values())} practitioner reports 
        about AI/LLM evaluation frameworks, create a detailed conceptual workflow that addresses identified 
        challenges and requirements with supporting evidence.
        
        KEY THEMES IDENTIFIED FROM PRACTITIONER CONTENT:
        {themes_text}
        
        CHALLENGE CATEGORIES WITH EVIDENCE:
        {challenges_summary}
        
        REQUIREMENTS WITH SOURCE ATTRIBUTION:
        {requirements_summary}
        
        Create a comprehensive conceptual workflow with 6-8 main steps that integrates evaluation into the 
        AI/LLM development lifecycle. For each step, provide:
        1. Step name and detailed description
        2. Specific challenges this step addresses (with evidence)
        3. Requirements this step fulfills (with source references)
        4. Implementation considerations from practitioner feedback
        5. Success metrics and validation approaches
        
        Format as:
        
        ## STEP [N]: [Step Name]
        
        **Description:** [Detailed 2-3 sentence description]
        
        **Addresses Challenges:** 
        - [Challenge category] (mentioned in [X] practitioner reports)
        - [Additional challenges as applicable]
        
        **Fulfills Requirements:**
        - [Requirement] (Source: [source type])
        - [Additional requirements as applicable]
        
        **Implementation Notes:** [Practical considerations from practitioner experiences]
        
        **Success Metrics:** [How to measure effectiveness of this step]
        
        ---
        
        Provide comprehensive workflow with evidence-based recommendations:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logging.error(f"Error synthesizing workflow: {e}")
            return "Error generating workflow synthesis - please check LLM connectivity"
    
    def _parse_themes_response(self, response: str) -> List[str]:
        """Parse LLM response to extract themes (legacy method)"""
        themes = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):  # Lines starting with numbers
                theme = re.sub(r'^\d+\.\s*', '', line)
                themes.append(theme)
        
        return themes
    
    def _parse_themes_with_evidence(self, response: str, source_info: Dict = None) -> List[Dict]:
        """Parse LLM response to extract themes with evidence and citations"""
        themes = []
        current_theme = {}
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('THEME:'):
                if current_theme:  # Save previous theme
                    if source_info:
                        current_theme['source'] = source_info.get('source', 'Unknown')
                        current_theme['source_id'] = source_info.get('id', 'N/A')
                        current_theme['source_url'] = source_info.get('url', '')
                        current_theme['source_title'] = source_info.get('title', '')
                    themes.append(current_theme)
                
                current_theme = {
                    'theme': line.replace('THEME:', '').strip(),
                    'description': '',
                    'evidence': '',
                    'confidence': 'Medium'
                }
            elif line.startswith('DESCRIPTION:'):
                current_theme['description'] = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('EVIDENCE:'):
                current_theme['evidence'] = line.replace('EVIDENCE:', '').strip().strip('"')
            elif line.startswith('CONFIDENCE:'):
                current_theme['confidence'] = line.replace('CONFIDENCE:', '').strip()
        
        # Add the last theme
        if current_theme and current_theme.get('theme'):  # Only add if has valid theme
            if source_info:
                current_theme['source'] = source_info.get('source', 'Unknown')
                current_theme['source_id'] = source_info.get('id', 'N/A')
                current_theme['source_url'] = source_info.get('url', '')
                current_theme['source_title'] = source_info.get('title', '')
            themes.append(current_theme)
        
        # Filter out any incomplete themes
        valid_themes = [t for t in themes if t.get('theme') and t['theme'].strip()]
        
        return valid_themes
    
    def _parse_categories_response(self, response: str) -> List[str]:
        """Parse LLM response to extract categories"""
        categories = []
        lines = response.strip().split('\n')
        
        valid_categories = [
            "Technical Challenges", "Integration Challenges", "Data Quality Challenges",
            "Scalability Challenges", "User Experience Challenges", "Performance Challenges"
        ]
        
        for line in lines:
            line = line.strip()
            if line in valid_categories:
                categories.append(line)
        
        return categories
    
    def _parse_requirements_response(self, response: str, source_doc: Dict) -> List[Dict]:
        """Parse LLM response to extract requirements"""
        requirements = []
        
        if "None" in response:
            return requirements
        
        sections = response.split("REQUIREMENT:")
        
        for section in sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            requirement_text = lines[0].strip()
            priority = "Medium"  # Default
            category = "Functional"  # Default
            
            for line in lines[1:]:
                if line.startswith("PRIORITY:"):
                    priority = line.replace("PRIORITY:", "").strip()
                elif line.startswith("CATEGORY:"):
                    category = line.replace("CATEGORY:", "").strip()
            
            requirements.append({
                "requirement": requirement_text,
                "priority": priority,
                "category": category,
                "source": source_doc['source'],
                "source_id": source_doc['id'],
                "source_url": source_doc['url']
            })
        
        return requirements

class TopicModeler:
    """Handles topic modeling using LDA"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['llm', 'model', 'ai', 'evaluation', 'framework'])  # Domain-specific stop words
    
    def extract_topics(self, documents: List[Dict], n_topics: int = 10) -> Dict[str, Any]:
        """Extract topics using Latent Dirichlet Allocation"""
        
        # Prepare text data
        texts = [doc['content'] for doc in documents]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        lda_matrix = lda.fit_transform(tfidf_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weight': topic.max()
            })
        
        # Assign documents to topics
        doc_topic_assignments = []
        for doc_idx, doc in enumerate(documents):
            topic_probs = lda_matrix[doc_idx]
            dominant_topic = topic_probs.argmax()
            doc_topic_assignments.append({
                'document_id': doc['id'],
                'dominant_topic': dominant_topic,
                'topic_probability': topic_probs[dominant_topic],
                'all_probabilities': topic_probs.tolist()
            })
        
        return {
            'topics': topics,
            'document_assignments': doc_topic_assignments,
            'model': lda,
            'vectorizer': vectorizer
        }

class FrequencyAnalyzer:
    """Analyzes word and phrase frequencies"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_frequencies(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze term frequencies across documents"""
        
        # Combine all text
        all_text = " ".join([doc['content'] for doc in documents])
        
        # Word frequency analysis
        words = word_tokenize(all_text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        word_freq = Counter(words)
        
        # Bi-gram analysis
        bigrams = list(nltk.bigrams(words))
        bigram_freq = Counter(bigrams)
        
        # Technical term frequency (domain-specific)
        technical_terms = [
            'evaluation metric', 'benchmark', 'testing framework', 'model performance',
            'validation', 'accuracy', 'precision', 'recall', 'f1 score', 'auc',
            'rag evaluation', 'llm testing', 'prompt engineering', 'fine tuning',
            'embedding', 'retrieval', 'generation', 'hallucination', 'bias detection'
        ]
        
        technical_freq = {}
        for term in technical_terms:
            count = all_text.lower().count(term.lower())
            if count > 0:
                technical_freq[term] = count
        
        return {
            'word_frequencies': dict(word_freq.most_common(50)),
            'bigram_frequencies': {f"{b[0]} {b[1]}": count for b, count in bigram_freq.most_common(30)},
            'technical_term_frequencies': technical_freq,
            'total_words': len(words),
            'unique_words': len(set(words))
        }

class ThematicAnalysisOrchestrator:
    """Orchestrates the complete thematic analysis process"""
    
    def __init__(self):
        self.llm_analyst = LLMAnalyst()
        self.topic_modeler = TopicModeler()
        self.frequency_analyzer = FrequencyAnalyzer()
        
        # Create output directories
        os.makedirs(OUTPUT_CONFIG["results_dir"], exist_ok=True)
    
    def analyze_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive thematic analysis"""
        logging.info(f"Starting thematic analysis on {len(documents)} documents")
        
        analysis_results = {
            'summary': {
                'total_documents': len(documents),
                'sources': Counter([doc['source'] for doc in documents]),
                'analysis_date': datetime.now().isoformat()
            }
        }
        
        # Frequency analysis
        logging.info("Performing frequency analysis...")
        freq_results = self.frequency_analyzer.analyze_frequencies(documents)
        analysis_results['frequency_analysis'] = freq_results
        
        # Topic modeling
        logging.info("Performing topic modeling...")
        topic_results = self.topic_modeler.extract_topics(
            documents, ANALYSIS_CONFIG['topic_modeling']['num_topics']
        )
        # Remove model objects for serialization
        analysis_results['topic_modeling'] = {
            'topics': topic_results['topics'],
            'document_assignments': topic_results['document_assignments']
        }
        
        # LLM-based thematic analysis with evidence
        logging.info("Extracting themes with evidence using LLM...")
        
        # Sample documents for LLM analysis (to avoid overwhelming the model)
        sample_docs = documents[:20] if len(documents) > 20 else documents
        
        all_themes_with_evidence = []
        for doc in sample_docs:
            source_info = {
                'source': doc['source'],
                'id': doc['id'],
                'url': doc['url'],
                'title': doc['title']
            }
            themes = self.llm_analyst.extract_themes_from_text(
                doc['content'], f"Source: {doc['source']}", source_info
            )
            all_themes_with_evidence.extend(themes)
        
        analysis_results['llm_themes_with_evidence'] = all_themes_with_evidence
        # Keep legacy format for backward compatibility - with error handling
        analysis_results['llm_themes'] = [
            theme.get('theme', 'Unknown Theme') for theme in all_themes_with_evidence 
            if isinstance(theme, dict) and theme.get('theme')
        ]
        
        # Challenge categorization
        logging.info("Categorizing challenges...")
        challenge_categories = self.llm_analyst.categorize_challenges(sample_docs)
        analysis_results['challenge_categories'] = {
            category: len(docs) for category, docs in challenge_categories.items()
        }
        
        # Requirements extraction
        logging.info("Extracting requirements...")
        requirements = self.llm_analyst.extract_requirements(sample_docs)
        analysis_results['requirements'] = requirements
        
        # Workflow synthesis with enhanced evidence
        logging.info("Synthesizing conceptual workflow...")
        # Pass themes with evidence if available, otherwise use simple theme names
        themes_for_workflow = all_themes_with_evidence if all_themes_with_evidence else analysis_results.get('llm_themes', [])
        
        # Fallback to basic themes if no LLM themes available
        if not themes_for_workflow:
            themes_for_workflow = ['AI/LLM Evaluation', 'Performance Metrics', 'Quality Assessment']
            logging.warning("No themes extracted, using fallback themes for workflow synthesis")
        
        workflow = self.llm_analyst.synthesize_workflow(
            themes_for_workflow, challenge_categories, requirements
        )
        analysis_results['conceptual_workflow'] = workflow
        
        # Generate insights for research questions
        analysis_results['research_question_insights'] = self._generate_rq_insights(
            analysis_results, documents
        )
        
        # Clean numpy types from results before saving
        analysis_results = self._clean_numpy_types(analysis_results)
        
        # Save results
        self._save_analysis_results(analysis_results)
        
        logging.info("Thematic analysis complete")
        return analysis_results
    
    def _generate_rq_insights(self, analysis_results: Dict, documents: List[Dict]) -> Dict[str, str]:
        """Generate comprehensive insights for each research question with evidence and citations"""
        insights = {}
        
        # Prepare evidence summaries with error handling
        themes_with_evidence = analysis_results.get('llm_themes_with_evidence', [])
        high_confidence_themes = [
            t for t in themes_with_evidence 
            if isinstance(t, dict) and t.get('confidence') == 'High' and t.get('theme')
        ]
        
        challenge_summary = "; ".join([
            f"{cat} ({count} reports)" 
            for cat, count in sorted(analysis_results['challenge_categories'].items(), 
                                   key=lambda x: x[1], reverse=True) if count > 0
        ][:5])
        
        requirements_by_priority = {}
        for req in analysis_results.get('requirements', []):
            priority = req.get('priority', 'Unknown')
            if priority not in requirements_by_priority:
                requirements_by_priority[priority] = []
            requirements_by_priority[priority].append(req)
        
        for i, rq in enumerate(RESEARCH_QUESTIONS, 1):
            # Create a comprehensive context for each research question
            evidence_context = ""
            
            if i == 1:  # Challenges question
                evidence_context = f"""
                Challenge Evidence Summary:
                - Primary challenge categories: {challenge_summary}
                - Supporting themes: {'; '.join([t['theme'] for t in high_confidence_themes[:3]])}
                - Sample practitioner quotes: {'; '.join([f'"{t["evidence"][:100]}..."' for t in high_confidence_themes[:2] if t.get('evidence')])}
                """
            elif i == 2:  # Requirements question
                high_priority_reqs = requirements_by_priority.get('High', [])[:3]
                evidence_context = f"""
                Requirements Evidence Summary:
                - High priority requirements: {'; '.join([req['requirement'][:80] for req in high_priority_reqs])}
                - Total requirements identified: {len(analysis_results.get('requirements', []))}
                - Source diversity: {len(set(req.get('source', 'Unknown') for req in analysis_results.get('requirements', [])))} different source types
                """
            elif i == 3:  # Workflow question
                evidence_context = f"""
                Workflow Synthesis Evidence:
                - Integration themes: {'; '.join([t['theme'] for t in themes_with_evidence[:4] if 'integration' in t.get('theme', '').lower() or 'workflow' in t.get('theme', '').lower()])}
                - Implementation challenges: {challenge_summary}
                - Practitioner-reported solutions: Available in {len(documents)} analyzed documents
                """
            
            prompt = f"""
            As a research analyst conducting a Multivocal Literature Review, provide a comprehensive, evidence-based 
            answer to this research question using the analyzed practitioner-generated content.
            
            **Research Question {i}:** {rq}
            
            **Analysis Foundation:**
            - Total documents analyzed: {len(documents)}
            - Data sources: {', '.join(analysis_results['summary']['sources'].keys())}
            - Analysis methods: Topic modeling, thematic analysis, frequency analysis
            
            {evidence_context}
            
            **Detailed Findings:**
            - Key themes identified: {', '.join(analysis_results['llm_themes'][:8])}
            - Most frequent technical terms: {', '.join(list(analysis_results.get('frequency_analysis', {}).get('technical_term_frequencies', {}).keys())[:6])}
            - Cross-source validation: Findings validated across multiple source types
            
            **Required Response Format:**
            
            ### Summary
            [2-3 sentence executive summary directly answering the research question]
            
            ### Key Findings
            [Bullet points of main findings with supporting evidence]
            
            ### Supporting Evidence
            [Specific examples, quotes, or data points from the analysis]
            
            ### Implications
            [What these findings mean for practitioners and researchers]
            
            ### Confidence Level
            [High/Medium/Low based on evidence quality and consistency]
            
            Provide a scholarly, evidence-based response that directly addresses the research question:
            """
            
            try:
                response = self.llm_analyst.llm.invoke(prompt)
                insights[f"RQ{i}"] = response
            except Exception as e:
                logging.error(f"Error generating insights for RQ{i}: {e}")
                insights[f"RQ{i}"] = f"Error generating insights: {e}"
        
        return insights
    
    def _clean_numpy_types(self, obj):
        """Recursively clean numpy types from data structures"""
        if isinstance(obj, dict):
            return {key: self._clean_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results as JSON
        json_file = os.path.join(
            OUTPUT_CONFIG["results_dir"], 
            f"thematic_analysis_{timestamp}.json"
        )
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # Save summary report
        report_file = os.path.join(
            OUTPUT_CONFIG["results_dir"], 
            f"analysis_report_{timestamp}.md"
        )
        self._generate_report(results, report_file)
        
        logging.info(f"Analysis results saved to {json_file} and {report_file}")
    
    def _generate_report(self, results: Dict[str, Any], output_file: str):
        """Generate a comprehensive markdown report with citations and evidence"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Title and metadata
            f.write("# Multivocal Literature Review: AI/LLM Evaluation Framework Analysis\n\n")
            f.write("## Executive Summary\n\n")
            f.write("This report presents a comprehensive analysis of practitioner-generated content ")
            f.write("regarding AI/LLM evaluation frameworks, synthesizing insights from GitHub repositories, ")
            f.write("Stack Overflow discussions, technical blogs, and Hugging Face resources.\n\n")
            
            f.write("## Methodology\n\n")
            f.write(f"**Analysis Date:** {results['summary']['analysis_date']}\n\n")
            f.write(f"**Total Documents Analyzed:** {results['summary']['total_documents']}\n\n")
            f.write("**Analysis Approach:** Mixed-methods combining automated topic modeling (LDA), ")
            f.write("frequency analysis, and LLM-assisted thematic analysis with human oversight.\n\n")
            
            # Data sources with detailed breakdown
            f.write("## Data Sources and Collection\n\n")
            total_docs = results['summary']['total_documents']
            f.write("| Source | Documents | Percentage | Description |\n")
            f.write("|--------|-----------|------------|-------------|\n")
            for source, count in results['summary']['sources'].items():
                percentage = (count / total_docs * 100) if total_docs > 0 else 0
                description = self._get_source_description(source)
                f.write(f"| {source} | {count} | {percentage:.1f}% | {description} |\n")
            f.write("\n")
            
            # Enhanced key themes with evidence
            f.write("## Key Themes and Findings\n\n")
            if 'llm_themes_with_evidence' in results:
                # Group themes by confidence level
                high_conf_themes = [t for t in results['llm_themes_with_evidence'] if t.get('confidence') == 'High']
                medium_conf_themes = [t for t in results['llm_themes_with_evidence'] if t.get('confidence') == 'Medium']
                
                f.write("### High-Confidence Themes\n\n")
                for i, theme in enumerate(high_conf_themes[:8], 1):
                    f.write(f"#### {i}. {theme['theme']}\n\n")
                    f.write(f"**Description:** {theme['description']}\n\n")
                    if theme.get('evidence'):
                        f.write(f"**Supporting Evidence:** \"{theme['evidence']}\"\n\n")
                    if theme.get('source_url'):
                        f.write(f"**Source:** [{theme.get('source', 'Unknown')}]({theme['source_url']})\n\n")
                    else:
                        f.write(f"**Source:** {theme.get('source', 'Unknown')} - {theme.get('source_id', 'N/A')}\n\n")
                
                if medium_conf_themes:
                    f.write("### Medium-Confidence Themes\n\n")
                    for theme in medium_conf_themes[:5]:
                        f.write(f"- **{theme['theme']}**: {theme['description']}")
                        if theme.get('source_url'):
                            f.write(f" ([Source]({theme['source_url']}))")
                        f.write("\n")
                    f.write("\n")
            else:
                # Fallback to legacy format
                for i, theme in enumerate(results['llm_themes'][:10], 1):
                    f.write(f"{i}. {theme}\n")
                f.write("\n")
            
            # Statistical analysis of challenges
            f.write("## Challenge Analysis\n\n")
            f.write("### Challenge Categories (Quantitative Analysis)\n\n")
            total_challenges = sum(count for count in results['challenge_categories'].values() if count > 0)
            
            f.write("| Challenge Category | Mentions | Percentage | Severity |\n")
            f.write("|--------------------|----------|------------|----------|\n")
            
            for category, count in sorted(results['challenge_categories'].items(), 
                                        key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / total_challenges * 100) if total_challenges > 0 else 0
                    severity = "High" if percentage > 30 else "Medium" if percentage > 15 else "Low"
                    f.write(f"| {category} | {count} | {percentage:.1f}% | {severity} |\n")
            f.write("\n")
            
            # Enhanced requirements analysis
            f.write("## Requirements Analysis\n\n")
            if results.get('requirements'):
                # Group by priority
                priority_groups = {}
                for req in results['requirements']:
                    priority = req.get('priority', 'Unknown')
                    if priority not in priority_groups:
                        priority_groups[priority] = []
                    priority_groups[priority].append(req)
                
                for priority in ['High', 'Medium', 'Low']:
                    if priority in priority_groups:
                        f.write(f"### {priority} Priority Requirements\n\n")
                        for req in priority_groups[priority][:10]:
                            f.write(f"- **{req['requirement']}**\n")
                            if req.get('source_url'):
                                f.write(f"  - *Source: [{req.get('source', 'Unknown')}]({req['source_url']})*\n")
                            else:
                                f.write(f"  - *Source: {req.get('source', 'Unknown')}*\n")
                        f.write("\n")
            
            # Topic modeling results
            if 'topic_modeling' in results and results['topic_modeling'].get('topics'):
                f.write("## Topic Modeling Results\n\n")
                f.write("### Discovered Topics (LDA Analysis)\n\n")
                for i, topic in enumerate(results['topic_modeling']['topics'][:8], 1):
                    f.write(f"**Topic {i}:** {', '.join(topic['words'][:5])}\n\n")
                f.write("\n")
            
            # Frequency analysis
            if 'frequency_analysis' in results:
                f.write("## Frequency Analysis\n\n")
                f.write("### Most Mentioned Technical Terms\n\n")
                tech_terms = results['frequency_analysis'].get('technical_term_frequencies', {})
                for term, count in sorted(tech_terms.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"- **{term}**: {count} mentions\n")
                f.write("\n")
            
            # Research question insights with enhanced formatting
            f.write("## Research Question Analysis\n\n")
            for rq_id, insight in results['research_question_insights'].items():
                rq_num = rq_id.replace('RQ', '')
                f.write(f"### Research Question {rq_num}\n\n")
                f.write(f"**Question:** {RESEARCH_QUESTIONS[int(rq_num)-1]}\n\n")
                f.write(f"**Analysis:** {insight}\n\n")
            
            # Conceptual workflow
            f.write("## Conceptual Framework\n\n")
            f.write("### Proposed Integration Workflow\n\n")
            f.write(results['conceptual_workflow'])
            f.write("\n\n")
            
            # Limitations and future work
            f.write("## Limitations and Future Research\n\n")
            f.write("### Study Limitations\n\n")
            f.write("- **Sample Bias**: Analysis limited to English-language, publicly available content\n")
            f.write("- **Temporal Scope**: Data collection reflects current state, may not capture emerging trends\n")
            f.write("- **Platform Bias**: Findings may be influenced by the specific platforms analyzed\n\n")
            
            f.write("### Future Research Directions\n\n")
            f.write("- Longitudinal analysis to track evolution of evaluation practices\n")
            f.write("- Cross-industry comparison of evaluation frameworks\n")
            f.write("- Empirical validation of identified requirements and challenges\n\n")
            
            # References section
            f.write("## References and Data Sources\n\n")
            self._write_references_section(f, results)
    
    def _get_source_description(self, source: str) -> str:
        """Get description for each data source"""
        descriptions = {
            'github_repository': 'Code repositories and documentation',
            'github_issue': 'Developer discussions and problem reports',
            'stackoverflow': 'Technical Q&A and community solutions',
            'web_article': 'Technical blogs and industry publications',
            'huggingface_model': 'Model cards and evaluation documentation',
            'huggingface_dataset': 'Dataset cards and benchmark descriptions',
            'huggingface_paper': 'Research papers and technical reports',
            'huggingface_blog': 'Technical blog posts and tutorials'
        }
        return descriptions.get(source, 'Mixed technical content')
    
    def _write_references_section(self, f, results: Dict[str, Any]):
        """Write a comprehensive references section"""
        f.write("### Primary Data Sources\n\n")
        
        # Collect unique sources from themes with evidence
        sources_cited = set()
        if 'llm_themes_with_evidence' in results:
            for theme in results['llm_themes_with_evidence']:
                if theme.get('source_url'):
                    source_key = f"{theme.get('source', 'Unknown')}|{theme.get('source_url', '')}"
                    sources_cited.add(source_key)
        
        # Add sources from requirements
        if results.get('requirements'):
            for req in results['requirements']:
                if req.get('source_url'):
                    source_key = f"{req.get('source', 'Unknown')}|{req.get('source_url', '')}"
                    sources_cited.add(source_key)
        
        # Write citations
        citation_num = 1
        for source_key in sorted(sources_cited):
            parts = source_key.split('|', 1)
            if len(parts) == 2:
                source_type, url = parts
                f.write(f"{citation_num}. {source_type}: {url}\n")
                citation_num += 1
        
        f.write(f"\n**Total Sources Analyzed:** {results['summary']['total_documents']}\n")
        f.write(f"**Analysis Methodology:** Mixed-methods MLR approach\n")
        f.write(f"**Data Collection Period:** {results['summary']['analysis_date']}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    analyzer = ThematicAnalysisOrchestrator()
    
    # Load processed documents
    processed_files = [f for f in os.listdir(OUTPUT_CONFIG["processed_data_dir"]) 
                      if f.startswith('processed_documents_') and f.endswith('.json')]
    
    if processed_files:
        latest_file = sorted(processed_files)[-1]
        file_path = os.path.join(OUTPUT_CONFIG["processed_data_dir"], latest_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        results = analyzer.analyze_documents(documents)
    else:
        logging.error("No processed documents found. Run data acquisition and preprocessing first.") 