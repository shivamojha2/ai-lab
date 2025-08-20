#!/usr/bin/env python3
"""
LLM Evaluation Pipeline - Practice

This script demonstrates production-ready ML evaluation practices:
- Data pipeline processing (JSONL/CSV)
- Scalable evaluation metrics computation  
- Edge case handling and data validation
- Clean CLI interface and logging
- Memory-efficient streaming processing

Task: Process LLM experiment logs and generate evaluation reports
Mirrors real production evaluation pipelines for LLM systems.
"""

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
import statistics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LLMEvaluationPipeline:
    """
    Production-ready LLM evaluation pipeline with comprehensive error handling.
    
    Features:
    - Streaming data processing for memory efficiency
    - Robust error handling and data validation
    - Multiple evaluation metrics (accuracy, BLEU-like, latency)
    - Experiment grouping and statistical analysis
    - Scalable design for production workloads
    """
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        """
        Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.processed_count = 0
        self.error_count = 0
        self.experiments = defaultdict(list)
        self.global_stats = {
            'total_tokens': 0,
            'total_latency': 0.0,
            'error_types': Counter()
        }
    
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single evaluation record with comprehensive checks.
        
        Args:
            record: Dictionary containing evaluation data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['id', 'input', 'output', 'reference']
        
        # Check required fields
        for field in required_fields:
            if field not in record:
                return False, f"Missing required field: {field}"
            
            # Check for None or empty values
            if record[field] is None or record[field] == "":
                return False, f"Empty value for required field: {field}"
        
        # Validate data types and ranges
        if 'score' in record:
            try:
                score = float(record['score'])
                if not (0.0 <= score <= 1.0):
                    return False, f"Score {score} not in valid range [0.0, 1.0]"
            except (ValueError, TypeError):
                return False, f"Invalid score format: {record.get('score')}"
        
        if 'latency_ms' in record:
            try:
                latency = float(record['latency_ms'])
                if latency < 0:
                    return False, f"Negative latency: {latency}"
                if latency > 300000:  # 5 minutes seems excessive
                    logger.warning(f"Very high latency detected: {latency}ms for record {record['id']}")
            except (ValueError, TypeError):
                return False, f"Invalid latency format: {record.get('latency_ms')}"
        
        # Validate text lengths (prevent memory issues)
        for text_field in ['input', 'output', 'reference']:
            if len(str(record[text_field])) > 50000:  # 50k chars max
                return False, f"Text too long in field {text_field}: {len(str(record[text_field]))} chars"
        
        return True, None
    
    def compute_text_similarity(self, predicted: str, reference: str) -> float:
        """
        Compute simple text similarity metric (simplified BLEU-like).
        
        Args:
            predicted: Model prediction
            reference: Ground truth reference
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not predicted or not reference:
            return 0.0
        
        # Simple word-level overlap (production would use proper BLEU/ROUGE)
        pred_words = set(str(predicted).lower().split())
        ref_words = set(str(reference).lower().split())
        
        if not ref_words:
            return 1.0 if not pred_words else 0.0
        
        overlap = len(pred_words & ref_words)
        return overlap / len(ref_words)
    
    def compute_exact_match(self, predicted: str, reference: str) -> float:
        """
        Compute exact match accuracy.
        
        Args:
            predicted: Model prediction
            reference: Ground truth reference
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if str(predicted).strip().lower() == str(reference).strip().lower() else 0.0
    
    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single evaluation record and compute metrics.
        
        Args:
            record: Raw evaluation record
            
        Returns:
            Processed record with computed metrics, or None if invalid
        """
        # Validate record
        is_valid, error_msg = self.validate_record(record)
        if not is_valid:
            self.error_count += 1
            self.global_stats['error_types'][error_msg] += 1
            logger.warning(f"Invalid record {record.get('id', 'unknown')}: {error_msg}")
            return None
        
        # Extract experiment info
        experiment_name = record.get('experiment', 'default')
        
        # Compute metrics
        processed = {
            'id': record['id'],
            'experiment': experiment_name,
            'input_length': len(str(record['input']).split()),
            'output_length': len(str(record['output']).split()),
            'exact_match': self.compute_exact_match(record['output'], record['reference']),
            'text_similarity': self.compute_text_similarity(record['output'], record['reference']),
        }
        
        # Add optional metrics
        if 'score' in record:
            processed['model_confidence'] = float(record['score'])
        
        if 'latency_ms' in record:
            processed['latency_ms'] = float(record['latency_ms'])
            self.global_stats['total_latency'] += processed['latency_ms']
        
        # Update global statistics
        self.global_stats['total_tokens'] += processed['input_length'] + processed['output_length']
        self.processed_count += 1
        
        # Group by experiment
        self.experiments[experiment_name].append(processed)
        
        return processed
    
    def stream_jsonl_file(self, filepath: Path) -> Iterator[Dict[str, Any]]:
        """
        Stream JSONL file line by line for memory efficiency.
        
        Args:
            filepath: Path to JSONL file
            
        Yields:
            Parsed JSON records
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        self.error_count += 1
                        continue
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            raise
    
    def stream_csv_file(self, filepath: Path) -> Iterator[Dict[str, Any]]:
        """
        Stream CSV file row by row for memory efficiency.
        
        Args:
            filepath: Path to CSV file
            
        Yields:
            Parsed CSV records as dictionaries
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, 1):
                    try:
                        yield row
                    except Exception as e:
                        logger.warning(f"Error processing row {row_num}: {e}")
                        self.error_count += 1
                        continue
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {e}")
            raise
    
    def process_file(self, filepath: Path) -> None:
        """
        Process evaluation file (JSONL or CSV) with streaming.
        
        Args:
            filepath: Path to evaluation file
        """
        logger.info(f"Processing file: {filepath}")
        
        # Determine file type and stream accordingly
        if filepath.suffix.lower() == '.jsonl':
            record_stream = self.stream_jsonl_file(filepath)
        elif filepath.suffix.lower() == '.csv':
            record_stream = self.stream_csv_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Process records
        batch_count = 0
        for record in record_stream:
            processed = self.process_record(record)
            if processed:
                batch_count += 1
                
                # Log progress periodically
                if batch_count % 1000 == 0:
                    logger.info(f"Processed {batch_count} records...")
        
        logger.info(f"Completed processing {filepath}: {batch_count} valid records")
    
    def compute_experiment_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute comprehensive statistics for each experiment.
        
        Returns:
            Dictionary with experiment statistics
        """
        experiment_stats = {}
        
        for exp_name, records in self.experiments.items():
            if not records:
                continue
            
            # Extract metrics
            exact_matches = [r['exact_match'] for r in records]
            similarities = [r['text_similarity'] for r in records]
            input_lengths = [r['input_length'] for r in records]
            output_lengths = [r['output_length'] for r in records]
            
            # Latency stats (if available)
            latencies = [r['latency_ms'] for r in records if 'latency_ms' in r]
            
            # Model confidence (if available)
            confidences = [r['model_confidence'] for r in records if 'model_confidence' in r]
            
            # Compute statistics with error handling
            def safe_stats(values: List[float]) -> Dict[str, float]:
                if not values:
                    return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
                
                return {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values)
                }
            
            stats = {
                'num_records': len(records),
                'exact_match': safe_stats(exact_matches),
                'text_similarity': safe_stats(similarities),
                'input_length': safe_stats(input_lengths),
                'output_length': safe_stats(output_lengths),
            }
            
            if latencies:
                stats['latency_ms'] = safe_stats(latencies)
                stats['throughput_tokens_per_sec'] = {
                    'mean': sum(r['output_length'] for r in records if 'latency_ms' in r) / 
                           (sum(latencies) / 1000) if sum(latencies) > 0 else 0.0
                }
            
            if confidences:
                stats['model_confidence'] = safe_stats(confidences)
            
            experiment_stats[exp_name] = stats
        
        return experiment_stats
    
    def generate_ranking(self, experiment_stats: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Generate experiment ranking based on composite score.
        
        Args:
            experiment_stats: Experiment statistics
            
        Returns:
            List of (experiment_name, composite_score) sorted by score
        """
        rankings = []
        
        for exp_name, stats in experiment_stats.items():
            # Composite score: weighted combination of metrics
            exact_match_score = stats['exact_match']['mean']
            similarity_score = stats['text_similarity']['mean']
            
            # Penalize high latency (if available)
            latency_penalty = 0.0
            if 'latency_ms' in stats and stats['latency_ms']['mean'] > 0:
                # Normalize latency penalty (assuming 1000ms is baseline)
                latency_penalty = min(0.2, stats['latency_ms']['mean'] / 5000.0)
            
            # Composite score (0.0 to 1.0)
            composite_score = (0.6 * exact_match_score + 
                             0.3 * similarity_score - 
                             0.1 * latency_penalty)
            
            rankings.append((exp_name, max(0.0, composite_score)))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def save_results(self, experiment_stats: Dict[str, Dict[str, Any]], 
                    rankings: List[Tuple[str, float]]) -> None:
        """
        Save evaluation results to files.
        
        Args:
            experiment_stats: Experiment statistics
            rankings: Experiment rankings
        """
        # Save detailed statistics (JSON)
        stats_file = self.output_dir / 'experiment_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_stats, f, indent=2, sort_keys=True)
        logger.info(f"Detailed statistics saved to: {stats_file}")
        
        # Save rankings (CSV)
        rankings_file = self.output_dir / 'experiment_rankings.csv'
        with open(rankings_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['experiment', 'composite_score', 'rank'])
            for rank, (exp_name, score) in enumerate(rankings, 1):
                writer.writerow([exp_name, f"{score:.4f}", rank])
        logger.info(f"Rankings saved to: {rankings_file}")
        
        # Save summary report
        summary_file = self.output_dir / 'evaluation_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("LLM Evaluation Pipeline - Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total records processed: {self.processed_count:,}\n")
            f.write(f"Total errors encountered: {self.error_count:,}\n")
            f.write(f"Total experiments: {len(experiment_stats)}\n")
            f.write(f"Total tokens processed: {self.global_stats['total_tokens']:,}\n")
            
            if self.global_stats['total_latency'] > 0:
                f.write(f"Total processing time: {self.global_stats['total_latency']:,.1f}ms\n")
            
            f.write(f"\nTop 5 Experiments by Performance:\n")
            f.write("-" * 30 + "\n")
            for rank, (exp_name, score) in enumerate(rankings[:5], 1):
                f.write(f"{rank}. {exp_name}: {score:.4f}\n")
            
            if self.error_count > 0:
                f.write(f"\nError Summary:\n")
                f.write("-" * 15 + "\n")
                for error_type, count in self.global_stats['error_types'].most_common():
                    f.write(f"{error_type}: {count}\n")
        
        logger.info(f"Summary report saved to: {summary_file}")
    
    def run_evaluation(self, input_files: List[Path]) -> None:
        """
        Run complete evaluation pipeline.
        
        Args:
            input_files: List of input files to process
        """
        logger.info(f"Starting LLM evaluation pipeline with {len(input_files)} files")
        
        # Process all input files
        for filepath in input_files:
            try:
                self.process_file(filepath)
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                continue
        
        # Compute statistics
        logger.info("Computing experiment statistics...")
        experiment_stats = self.compute_experiment_statistics()
        
        # Generate rankings
        logger.info("Generating experiment rankings...")
        rankings = self.generate_ranking(experiment_stats)
        
        # Save results
        logger.info("Saving evaluation results...")
        self.save_results(experiment_stats, rankings)
        
        # Log final summary
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Processed: {self.processed_count:,} records")
        logger.info(f"Errors: {self.error_count:,} records")
        logger.info(f"Experiments: {len(experiment_stats)}")


def create_sample_data(output_file: Path, num_records: int = 1000) -> None:
    """
    Create sample evaluation data for testing.
    
    Args:
        output_file: Path to output file
        num_records: Number of sample records to generate
    """
    import random
    random.seed(42)
    
    experiments = ['baseline', 'fine_tuned_v1', 'fine_tuned_v2', 'rlhf_model']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_records):
            experiment = random.choice(experiments)
            
            # Simulate different performance levels per experiment
            if experiment == 'baseline':
                score = random.uniform(0.3, 0.6)
                latency = random.uniform(800, 1500)
            elif experiment == 'rlhf_model':
                score = random.uniform(0.8, 0.95)
                latency = random.uniform(1200, 2000)
            else:
                score = random.uniform(0.6, 0.8)
                latency = random.uniform(600, 1200)
            
            # Add some invalid records to test error handling
            if random.random() < 0.02:  # 2% invalid records
                record = {
                    'id': f'invalid_{i}',
                    'experiment': experiment,
                    'input': '',  # Invalid empty input
                    'output': 'some output',
                    'reference': 'some reference'
                }
            else:
                record = {
                    'id': f'eval_{i:06d}',
                    'experiment': experiment,
                    'input': f'What is the capital of country {i}?',
                    'output': f'The capital is city_{i}',
                    'reference': f'The capital is city_{i}' if random.random() < score else f'Different answer_{i}',
                    'score': score,
                    'latency_ms': latency
                }
            
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Generated {num_records} sample records in {output_file}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='LLM Evaluation Pipeline - Process experiment logs and generate evaluation reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input experiments.jsonl --output ./results
  %(prog)s --input exp1.jsonl exp2.jsonl --output ./eval_results
  %(prog)s --generate-sample --sample-file sample_data.jsonl
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        type=Path,
        help='Input evaluation files (JSONL or CSV format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./evaluation_results',
        help='Output directory for results (default: ./evaluation_results)'
    )
    
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate sample data for testing'
    )
    
    parser.add_argument(
        '--sample-file',
        type=Path,
        default='sample_evaluation_data.jsonl',
        help='Output file for sample data (default: sample_evaluation_data.jsonl)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of sample records to generate (default: 1000)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.generate_sample:
            logger.info("Generating sample evaluation data...")
            create_sample_data(args.sample_file, args.sample_size)
            logger.info(f"Sample data generated: {args.sample_file}")
            return
        
        if not args.input:
            logger.error("No input files specified. Use --input or --generate-sample")
            sys.exit(1)
        
        # Validate input files
        for filepath in args.input:
            if not filepath.exists():
                logger.error(f"Input file not found: {filepath}")
                sys.exit(1)
        
        # Run evaluation pipeline
        pipeline = LLMEvaluationPipeline(output_dir=args.output)
        pipeline.run_evaluation(args.input)
        
        logger.info("Evaluation pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
