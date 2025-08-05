"""
MLR System - Main Orchestration Script
Multivocal Literature Review System for AI/LLM Evaluation Framework Research
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_acquisition import DataAcquisitionOrchestrator
from modules.filtering_preprocessing import FilteringPreprocessingOrchestrator
from modules.thematic_analysis import ThematicAnalysisOrchestrator
from config import OUTPUT_CONFIG, RESEARCH_QUESTIONS

class MLRSystem:
    """Main MLR System orchestrator"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.setup_logging()
        self.create_directories()
        
        # Initialize components
        self.data_acquisition = DataAcquisitionOrchestrator(github_token)
        self.preprocessing = FilteringPreprocessingOrchestrator()
        self.analysis = ThematicAnalysisOrchestrator()
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(OUTPUT_CONFIG["logs_dir"], exist_ok=True)
        
        log_file = os.path.join(
            OUTPUT_CONFIG["logs_dir"], 
            f"mlr_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary output directories"""
        for dir_path in OUTPUT_CONFIG.values():
            if isinstance(dir_path, str):
                os.makedirs(dir_path, exist_ok=True)
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete MLR pipeline"""
        self.logger.info("="*60)
        self.logger.info("Starting MLR System - Complete Pipeline")
        self.logger.info("="*60)
        
        pipeline_start = datetime.now()
        results = {}
        
        try:
            # Phase 1: Data Acquisition
            self.logger.info("\n" + "="*40)
            self.logger.info("PHASE 1: DATA ACQUISITION")
            self.logger.info("="*40)
            
            acquisition_start = datetime.now()
            raw_data = self.data_acquisition.collect_all_data()
            acquisition_time = (datetime.now() - acquisition_start).total_seconds()
            
            results['data_acquisition'] = {
                'status': 'completed',
                'duration_seconds': acquisition_time,
                'data_summary': raw_data
            }
            
            self.logger.info(f"Data acquisition completed in {acquisition_time:.2f} seconds")
            
            # Phase 2: Filtering and Preprocessing
            self.logger.info("\n" + "="*40)
            self.logger.info("PHASE 2: FILTERING AND PREPROCESSING")
            self.logger.info("="*40)
            
            preprocessing_start = datetime.now()
            processed_documents = self.preprocessing.process_all_data()
            preprocessing_time = (datetime.now() - preprocessing_start).total_seconds()
            
            results['preprocessing'] = {
                'status': 'completed',
                'duration_seconds': preprocessing_time,
                'documents_count': len(processed_documents)
            }
            
            self.logger.info(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
            self.logger.info(f"Final document count: {len(processed_documents)}")
            
            # Phase 3: Thematic Analysis
            self.logger.info("\n" + "="*40)
            self.logger.info("PHASE 3: THEMATIC ANALYSIS")
            self.logger.info("="*40)
            
            analysis_start = datetime.now()
            analysis_results = self.analysis.analyze_documents(processed_documents)
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            
            results['thematic_analysis'] = {
                'status': 'completed',
                'duration_seconds': analysis_time,
                'analysis_summary': analysis_results['summary']
            }
            
            self.logger.info(f"Thematic analysis completed in {analysis_time:.2f} seconds")
            
            # Overall results
            total_time = (datetime.now() - pipeline_start).total_seconds()
            results['pipeline_summary'] = {
                'status': 'completed',
                'total_duration_seconds': total_time,
                'total_duration_minutes': total_time / 60,
                'phases_completed': 3,
                'completion_time': datetime.now().isoformat()
            }
            
            # Set top-level status for main() function check
            results['status'] = 'completed'
            
            self.logger.info("\n" + "="*60)
            self.logger.info("MLR PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            self.logger.info(f"Total execution time: {total_time/60:.2f} minutes")
            self.logger.info(f"Documents processed: {len(processed_documents)}")
            self.logger.info(f"Research questions addressed: {len(RESEARCH_QUESTIONS)}")
            
            # Print key findings
            self._print_key_findings(analysis_results)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            # Don't raise - return results so main() can handle it properly
        
        return results
    
    def run_data_acquisition_only(self) -> dict:
        """Run only the data acquisition phase"""
        self.logger.info("Running data acquisition only...")
        
        try:
            raw_data = self.data_acquisition.collect_all_data()
            self.logger.info("Data acquisition completed successfully")
            return {'status': 'completed', 'data': raw_data}
        except Exception as e:
            self.logger.error(f"Data acquisition failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_preprocessing_only(self) -> dict:
        """Run only the preprocessing phase"""
        self.logger.info("Running preprocessing only...")
        
        try:
            processed_documents = self.preprocessing.process_all_data()
            self.logger.info(f"Preprocessing completed. Documents: {len(processed_documents)}")
            return {'status': 'completed', 'documents_count': len(processed_documents)}
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_analysis_only(self) -> dict:
        """Run only the thematic analysis phase"""
        self.logger.info("Running thematic analysis only...")
        
        try:
            # Load latest processed documents
            import json
            processed_files = [
                f for f in os.listdir(OUTPUT_CONFIG["processed_data_dir"]) 
                if f.startswith('processed_documents_') and f.endswith('.json')
            ]
            
            if not processed_files:
                raise FileNotFoundError("No processed documents found. Run preprocessing first.")
            
            latest_file = sorted(processed_files)[-1]
            file_path = os.path.join(OUTPUT_CONFIG["processed_data_dir"], latest_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            analysis_results = self.analysis.analyze_documents(documents)
            self.logger.info("Thematic analysis completed successfully")
            
            return {'status': 'completed', 'results': analysis_results['summary']}
            
        except Exception as e:
            self.logger.error(f"Thematic analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _print_key_findings(self, analysis_results: dict):
        """Print key findings summary"""
        self.logger.info("\n" + "="*50)
        self.logger.info("KEY FINDINGS SUMMARY")
        self.logger.info("="*50)
        
        # Top themes
        if 'llm_themes' in analysis_results:
            self.logger.info("\nTop Themes Identified:")
            for i, theme in enumerate(analysis_results['llm_themes'][:5], 1):
                self.logger.info(f"  {i}. {theme}")
        
        # Challenge categories
        if 'challenge_categories' in analysis_results:
            self.logger.info("\nChallenge Categories:")
            for category, count in analysis_results['challenge_categories'].items():
                if count > 0:
                    self.logger.info(f"  - {category}: {count} mentions")
        
        # Requirements count
        if 'requirements' in analysis_results:
            self.logger.info(f"\nTotal Requirements Extracted: {len(analysis_results['requirements'])}")
            
            # Group by priority
            priority_counts = {}
            for req in analysis_results['requirements']:
                priority = req.get('priority', 'Unknown')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            for priority, count in priority_counts.items():
                self.logger.info(f"  - {priority} Priority: {count}")
        
        self.logger.info("\n" + "="*50)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MLR System for AI/LLM Evaluation Framework Research"
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'acquisition', 'preprocessing', 'analysis'],
        default='full',
        help='Run mode: full pipeline or specific phase'
    )
    
    parser.add_argument(
        '--github-token',
        type=str,
        help='GitHub API token for enhanced rate limits'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    mlr_system = MLRSystem(github_token=args.github_token)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run based on mode
    try:
        if args.mode == 'full':
            results = mlr_system.run_complete_pipeline()
        elif args.mode == 'acquisition':
            results = mlr_system.run_data_acquisition_only()
        elif args.mode == 'preprocessing':
            results = mlr_system.run_preprocessing_only()
        elif args.mode == 'analysis':
            results = mlr_system.run_analysis_only()
        
        if results.get('status') == 'completed':
            print("\n🎉 MLR System completed successfully!")
            print(f"📊 Check the results in: {OUTPUT_CONFIG['results_dir']}")
            print(f"📝 Logs available in: {OUTPUT_CONFIG['logs_dir']}")
        else:
            print(f"\n❌ MLR System failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  MLR System interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 MLR System crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 