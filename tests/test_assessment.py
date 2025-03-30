# tests/test_issues_assessment.py
import os
import sys
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import project components
from core.models.document import Document
from core.models.context import ProcessingContext
from core.llm.customllm import CustomLLM
from orchestrator import Orchestrator
from assessments.loader import AssessmentLoader

def progress_callback(progress: float, message: str) -> None:
    """Simple progress display callback."""
    percent = int(progress * 100)
    bar_length = 40
    filled_length = int(bar_length * progress)
    
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\r[{bar}] {percent}% {message}", end='', flush=True)
    
    if progress >= 1.0:
        print()

async def test_planner(document, llm, assessment_config):
    """Test just the planner stage."""
    logger.info("Testing planner stage...")
    
    # Create a processing context
    context = ProcessingContext(document.text, {
        "assessment_type": "issues",
        "assessment_config": assessment_config
    })
    
    # Load planner agent
    from core.agents.planner import PlannerAgent
    planner = PlannerAgent(llm)
    
    # Process with planner
    planning_result = await planner.process(context)
    
    # Save planning result
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"planner_result_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(planning_result, f, indent=2)
    
    logger.info(f"Planner results saved to {output_path}")
    return planning_result

async def test_full_pipeline(document, api_key, model="gpt-4", assessment_type="issues"):
    """Test the full agent pipeline with orchestrator."""
    logger.info(f"Testing full pipeline for {assessment_type} assessment...")
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize orchestrator with options
    orchestrator = Orchestrator(
        assessment_type=assessment_type,
        options={
            "api_key": api_key,
            "model": model
        }
    )
    
    # Process document
    result = await orchestrator.process_with_progress(document, progress_callback)
    
    # Save results
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{assessment_type}_result_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Full pipeline results saved to {output_path}")
    return result

async def main():
    """Run test for issues assessment."""
    try:
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
            return
        
        # Initialize LLM
        model = "gpt-4"
        llm = CustomLLM(api_key, model=model)
        logger.info(f"Initialized LLM with model: {model}")
        
        # Load assessment config
        assessment_loader = AssessmentLoader()
        assessment_config = assessment_loader.load_assessment("issues")
        if not assessment_config:
            logger.error("Failed to load issues assessment configuration")
            return
        
        # Load sample document
        sample_path = project_root / "data" / "samples" / "test.txt"
        if not sample_path.exists():
            logger.error(f"Sample file not found at {sample_path}")
            return
        
        document = Document.from_file(str(sample_path))
        logger.info(f"Loaded document: {document.filename}, {document.word_count} words")
        
        # First test just the planner
        planning_result = await test_planner(document, llm, assessment_config)
        print("\nPlanner Result Preview:")
        print(json.dumps(planning_result, indent=2)[:500] + "...\n")
        
        # Ask if user wants to continue with full pipeline
        response = input("Continue with full pipeline? (y/n): ").strip().lower()
        if response == 'y':
            full_result = await test_full_pipeline(document, api_key, model)
            
            # Show brief summary of results
            if "result" in full_result and "statistics" in full_result.get("result", {}):
                stats = full_result.get("result", {}).get("statistics", {})
                print("\nResults Summary:")
                print(f"- Total issues found: {stats.get('total_issues', 'N/A')}")
                print(f"- Processing time: {full_result.get('metadata', {}).get('processing_time', 0):.2f} seconds")
            else:
                print("\nFull pipeline complete. See the output file for details.")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
