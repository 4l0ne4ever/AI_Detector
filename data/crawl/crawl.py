#!/usr/bin/env python3
"""
Efficient arXiv-only crawler for collecting scientific papers.
This crawler focuses on arXiv API for reliable and fast data collection.
"""

import os
import json
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlencode

# Configuration
TARGET_SAMPLES = 50000
OUTPUT_FILE = "data/human_text_50k.jsonl"
MIN_TEXT_LENGTH = 200

# Expanded arXiv categories for maximum diversity
ARXIV_CATEGORIES = [
    # Computer Science
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.IR", "cs.HC", "cs.RO",
    "cs.CR", "cs.DB", "cs.DC", "cs.SE", "cs.SI", "cs.SY", "cs.AR", "cs.CC",
    "cs.CE", "cs.CG", "cs.GT", "cs.IT", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
    "cs.NA", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.SC", "cs.SD",
    # Statistics & Math
    "stat.ML", "stat.AP", "stat.CO", "stat.ME", "stat.TH", 
    "math.OC", "math.ST", "math.PR", "math.NA", "math.CO", "math.IT",
    # Physics & Related
    "physics.data-an", "physics.comp-ph", "physics.bio-ph", "physics.med-ph",
    "physics.soc-ph", "cond-mat.stat-mech", "cond-mat.dis-nn",
    # Biology & Quantitative
    "q-bio.QM", "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC",
    "q-bio.PE", "q-bio.SC", "q-bio.TO",
    # Economics & Finance
    "econ.EM", "econ.GN", "econ.TH", "q-fin.CP", "q-fin.EC", "q-fin.GN",
    "q-fin.MF", "q-fin.PM", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR"
]

# Setup basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add file handler after ensuring directory exists
def setup_file_logging():
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler('logs/arxiv_crawl.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def fetch_arxiv_papers(category, start=0, max_results=100, sort_by='submittedDate'):
    """Fetch papers from arXiv API for a specific category."""
    base_url = "http://export.arxiv.org/api/query?"
    
    params = {
        'search_query': f'cat:{category}',
        'start': start,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': 'descending'
    }
    
    url = base_url + urlencode(params)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from arXiv for category {category}: {e}")
        return None

def parse_arxiv_xml(xml_content):
    """Parse arXiv XML response and extract paper information."""
    papers = []
    
    try:
        # Handle namespace
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_content)
        
        for entry in root.findall('atom:entry', namespace):
            try:
                # Extract title
                title_elem = entry.find('atom:title', namespace)
                title = title_elem.text.strip() if title_elem is not None else ""
                
                # Extract abstract
                summary_elem = entry.find('atom:summary', namespace)
                abstract = summary_elem.text.strip() if summary_elem is not None else ""
                
                # Extract arXiv ID
                id_elem = entry.find('atom:id', namespace)
                arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None else ""
                
                # Extract published date
                published_elem = entry.find('atom:published', namespace)
                published = published_elem.text if published_elem is not None else ""
                
                # Extract categories
                categories = []
                for cat_elem in entry.findall('atom:category', namespace):
                    cat_term = cat_elem.get('term', '')
                    if cat_term:
                        categories.append(cat_term)
                
                # Extract authors
                authors = []
                for author_elem in entry.findall('atom:author', namespace):
                    name_elem = author_elem.find('atom:name', namespace)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                
                if title and abstract:
                    papers.append({
                        'arxiv_id': arxiv_id,
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'categories': categories,
                        'published': published,
                        'source': 'arxiv'
                    })
                    
            except Exception as e:
                logger.warning(f"Error parsing individual paper entry: {e}")
                continue
                
    except ET.ParseError as e:
        logger.error(f"Error parsing XML: {e}")
    
    return papers

def process_paper(paper):
    """Process a paper and return formatted data for output."""
    # Combine title and abstract
    text = f"{paper['title']}\n\n{paper['abstract']}"
    
    # Filter out very short texts
    if len(text) < MIN_TEXT_LENGTH:
        return None
    
    return {
        'text': text,
        'source': paper['source'],
        'metadata': {
            'arxiv_id': paper.get('arxiv_id', ''),
            'title': paper['title'],
            'authors': paper.get('authors', []),
            'categories': paper.get('categories', []),
            'published': paper.get('published', ''),
            'text_length': len(text)
        }
    }

def load_existing_papers():
    """Load existing papers to avoid duplicates."""
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'metadata' in data and 'arxiv_id' in data['metadata']:
                            existing_ids.add(data['metadata']['arxiv_id'])
        except Exception as e:
            logger.warning(f"Error loading existing papers: {e}")
    
    logger.info(f"Found {len(existing_ids)} existing papers")
    return existing_ids

def save_papers(papers, output_file):
    """Save papers to JSONL file."""
    saved_count = 0
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for paper in papers:
            try:
                json.dump(paper, f, ensure_ascii=False)
                f.write('\n')
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving paper: {e}")
    
    return saved_count

def crawl_arxiv():
    """Main crawling function."""
    ensure_directories()
    setup_file_logging()
    
    logger.info(f"Starting arXiv crawl - Target: {TARGET_SAMPLES} samples")
    logger.info(f"Output file: {OUTPUT_FILE}")
    
    # Load existing papers to avoid duplicates
    existing_ids = load_existing_papers()
    
    total_processed = 0
    total_saved = 0
    start_time = time.time()
    
    try:
        for category in ARXIV_CATEGORIES:
            logger.info(f"Processing category: {category}")
            
            # Try different sorting methods for more diversity
            sort_methods = ['submittedDate', 'lastUpdatedDate', 'relevance']
            
            for sort_method in sort_methods:
                # Fetch multiple batches per category per sort method
                for batch_start in range(0, 500, 100):  # Reduced batches per sort to get more variety
                    logger.info(f"Fetching batch {batch_start//100 + 1} for {category} sorted by {sort_method} (start={batch_start})")
                    
                    # Fetch papers from arXiv
                    xml_content = fetch_arxiv_papers(category, start=batch_start, max_results=100, sort_by=sort_method)
                    if not xml_content:
                        logger.warning(f"No content received for {category}, batch {batch_start}, sort: {sort_method}")
                        continue
                
                    # Parse XML
                    papers = parse_arxiv_xml(xml_content)
                    if not papers:
                        logger.warning(f"No papers parsed for {category}, batch {batch_start}, sort: {sort_method}")
                        break  # No more papers for this sort method
                
                    # Process and filter papers
                    processed_papers = []
                    for paper in papers:
                        # Skip if already exists
                        if paper.get('arxiv_id') in existing_ids:
                            continue
                        
                        processed_paper = process_paper(paper)
                        if processed_paper:
                            processed_papers.append(processed_paper)
                            existing_ids.add(paper.get('arxiv_id', ''))
                    
                    total_processed += len(papers)
                    
                    if processed_papers:
                        saved_count = save_papers(processed_papers, OUTPUT_FILE)
                        total_saved += saved_count
                        logger.info(f"Saved {saved_count} new papers from {category} (sort: {sort_method})")
                    
                    # Progress update
                    elapsed_time = time.time() - start_time
                    logger.info(f"Progress: {total_saved}/{TARGET_SAMPLES} papers saved "
                              f"({total_processed} processed, {elapsed_time:.1f}s elapsed)")
                    
                    # Check if we have enough samples
                    if total_saved >= TARGET_SAMPLES:
                        logger.info(f"Target reached! {total_saved} papers saved.")
                        return
                    
                    # Rate limiting - be respectful to arXiv
                    time.sleep(1)
        
        logger.info(f"Crawling completed. Total papers saved: {total_saved}")
        
    except KeyboardInterrupt:
        logger.info("Crawling interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during crawling: {e}")
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Final stats: {total_saved} papers saved, {total_processed} processed, "
                   f"{elapsed_time:.1f}s elapsed")

if __name__ == "__main__":
    crawl_arxiv()