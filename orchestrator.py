# orchestrator.py
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union

from core.models.document import Document
from core.models.context import ProcessingContext
from core.llm.customllm import CustomLLM
from utils.chunking import chunk_document

class Orchestrator:
    """
    Coordinates the document processing pipeline.
    Enhanced implementation with summarization and robust error handling.
    """
    
    def __init__(self, assessment_type: str, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the orchestrator.
        
        Args:
            assessment_type: Type of assessment to perform (e.g., "issues", "action_items")
            options: Configuration options for processing
        """
        self.assessment_type = assessment_type
        self.options = options or {}
        self.context = None
        
        # Initialize LLM if API key is provided
        self.llm = None
        if "api_key" in self.options:
            self.llm = CustomLLM(self.options["api_key"], model=self.options.get("model", "gpt-3.5-turbo"))
    
    async def process_document(self, 
                               document: Document,
                               progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Process a document through the pipeline.
        
        Args:
            document: Document object to process
            progress_callback: Optional callback for progress updates
                
        Returns:
            Processing results as a dictionary
        """
        # Create context using document
        self.context = ProcessingContext(document.text, self.options)
        
        if progress_callback:
            self.context.set_progress_callback(progress_callback)
        
        try:
            # Document analysis
            await self._analyze_document(document)
            
            # Chunking
            await self._chunk_document(document)
            
            # If LLM is initialized, perform summarization
            if self.llm:
                # Summarize chunks
                await self._summarize_chunks()
                
                # Combine summaries
                await self._combine_summaries(document)
            
            # Return the final result
            return self.context.get_final_result()
            
        except Exception as e:
            # Handle any uncaught exceptions
            if self.context.metadata["current_stage"]:
                self.context.fail_stage(self.context.metadata["current_stage"], str(e))
            else:
                self.context.fail_stage("unknown", str(e))
            
            # Return partial results with error information
            return self.context.get_final_result()
    
    async def _analyze_document(self, document: Document) -> None:
        """Analyze document to extract basic metadata."""
        self.context.set_stage("document_analysis")
        
        try:
            # Get document info from our Document object
            document_info = document.get_summary()
            
            # Store in context
            self.context.document_info = document_info
            
            # Simulate a bit of processing time for demo purposes
            await asyncio.sleep(0.1)
            
            self.context.complete_stage("document_analysis", document_info)
            
        except Exception as e:
            self.context.fail_stage("document_analysis", str(e))
            raise
    
    async def _chunk_document(self, document: Document) -> None:
        """Split document into processable chunks."""
        self.context.set_stage("chunking")
        
        try:
            # Get chunking parameters from options with defaults
            chunk_size = self.options.get("chunk_size", 10000)
            chunk_overlap = self.options.get("chunk_overlap", 500)
            
            # Perform chunking
            chunks = chunk_document(
                document,
                target_chunk_size=chunk_size,
                overlap=chunk_overlap
            )
            
            # Store chunks in context
            self.context.set_chunks(chunks)
            
            # Update progress as we process
            for i, _ in enumerate(chunks):
                progress = (i + 1) / len(chunks)
                self.context.update_stage_progress(
                    progress, f"Processed chunk {i+1} of {len(chunks)}"
                )
                # Small delay to make progress visible in UI
                await asyncio.sleep(0.01)
            
            # Complete the chunking stage
            chunk_info = {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(c.get("word_count", 0) for c in chunks) / max(len(chunks), 1),
                "chunking_strategy": f"size-{chunk_size}-overlap-{chunk_overlap}"
            }
            
            self.context.complete_stage("chunking", chunk_info)
            
        except Exception as e:
            self.context.fail_stage("chunking", str(e))
            raise
    
    async def _summarize_chunks(self) -> None:
        """Summarize document chunks in parallel."""
        self.context.set_stage("summarization")
        
        try:
            chunks = self.context.chunks
            
            if not chunks:
                self.context.add_warning("No chunks available for summarization")
                self.context.complete_stage("summarization", {"message": "No chunks to summarize"})
                return
            
            # Create tasks for all chunks (will run in batches)
            chunk_summaries = []
            
            # Process chunks in batches of 3 to avoid rate limits
            batch_size = 3
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                
                # Create batch of tasks
                batch_tasks = [
                    self._summarize_chunk(chunk, i) 
                    for i, chunk in enumerate(chunks[start_idx:end_idx], start=start_idx)
                ]
                
                # Process batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results
                for i, result in enumerate(batch_results):
                    chunk_idx = start_idx + i
                    
                    # Handle exceptions
                    if isinstance(result, Exception):
                        self.context.add_warning(f"Error summarizing chunk {chunk_idx}: {str(result)}")
                        chunk_summaries.append({
                            "chunk_index": chunk_idx,
                            "summary": f"Error: {str(result)}",
                            "error": str(result)
                        })
                    else:
                        chunk_summaries.append(result)
                
                # Update progress
                progress = (batch_idx + 1) / total_batches
                self.context.update_stage_progress(
                    progress, f"Summarized batch {batch_idx+1}/{total_batches} ({end_idx}/{len(chunks)} chunks)"
                )
            
            # Store summaries in context
            self.context.complete_stage("summarization", {
                "chunk_summaries": chunk_summaries,
                "total_chunks_summarized": len(chunk_summaries),
                "successful_summaries": sum(1 for s in chunk_summaries if "error" not in s)
            })
            
        except Exception as e:
            self.context.fail_stage("summarization", str(e))
            raise
    
    async def _summarize_chunk(self, chunk: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """Summarize a single chunk."""
        # Create a prompt for summarization
        prompt = f"""
        Please summarize the following text chunk ({chunk_index+1}):
        
        {chunk['text']}
        
        Provide a concise summary that captures the key points, maintaining any important details, 
        names, numbers, or findings. The summary should be about 1/4 the length of the original.
        """
        
        # Call LLM
        summary = await self.llm.generate_completion(prompt, max_tokens=500, temperature=0.3)
        
        return {
            "chunk_index": chunk_index,
            "summary": summary,
            "word_count": len(summary.split()),
            "original_word_count": chunk['word_count']
        }
    
    async def _combine_summaries(self, document: Document) -> None:
        """Combine chunk summaries into a coherent final summary."""
        self.context.set_stage("combination")
        
        try:
            # Get chunk summaries from the previous stage
            summarization_result = self.context.results.get("summarization", {})
            chunk_summaries = summarization_result.get("chunk_summaries", [])
            
            if not chunk_summaries:
                self.context.add_warning("No chunk summaries available to combine")
                self.context.complete_stage("combination", {"message": "No summaries to combine"})
                return
            
            # Sort summaries by chunk index
            sorted_summaries = sorted(chunk_summaries, key=lambda x: x.get("chunk_index", 0))
            
            # Extract the summary texts
            summary_texts = [s.get("summary", "") for s in sorted_summaries if "error" not in s]
            combined_text = "\n\n".join(summary_texts)
            
            # If there's nothing to combine, return an error
            if not combined_text.strip():
                self.context.add_warning("No valid summaries to combine")
                self.context.complete_stage("combination", {"message": "No valid summaries to combine"})
                return
            
            # Create prompt for the combined summary
            prompt = f"""
            Below are summaries of different sections of a document titled "{document.filename or 'Document'}".
            
            {combined_text}
            
            Please create a single coherent summary that combines all of these sections.
            Organize the information logically, eliminate redundancies, and ensure smooth transitions.
            The final summary should provide a comprehensive overview of the entire document, 
            highlighting the most important points, findings, or conclusions.
            """
            
            # Call LLM
            final_summary = await self.llm.generate_completion(prompt, max_tokens=1000, temperature=0.3)
            
            # Complete the stage with the final summary
            self.context.complete_stage("combination", {
                "final_summary": final_summary,
                "summary_word_count": len(final_summary.split()),
                "original_word_count": document.word_count,
                "reduction_percentage": (1 - len(final_summary.split()) / document.word_count) * 100
            })
            
        except Exception as e:
            self.context.fail_stage("combination", str(e))
            raise
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        if not self.context:
            return {
                "progress": 0.0,
                "message": "Not started",
                "current_stage": None
            }
        
        return {
            "progress": self.context.metadata["progress"],
            "message": self.context.metadata["progress_message"],
            "current_stage": self.context.metadata["current_stage"],
            "stages": self.context.metadata["stages"]
        }

    def update_progress(self, progress: float, message: str) -> None:
        """
        Update progress information.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: Progress message
        """
        if self.context:
            self.context.update_progress(progress, message)
        else:
            # Log progress even if context isn't initialized
            import logging
            logging.info(f"Progress: {progress:.2f} - {message}")

    async def process_with_progress(self, 
                                document: Document,
                                progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Process a document with detailed progress reporting.
        
        Args:
            document: Document object to process
            progress_callback: Optional callback for progress updates
                
        Returns:
            Processing results as a dictionary
        """
        # Start timing
        start_time = time.time()
        
        # Initialize progress
        if progress_callback:
            progress_callback(0.0, "Starting document processing")
        
        # Process document
        result = await self.process_document(document, progress_callback)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0, f"Processing complete in {processing_time:.2f} seconds")
        
        # Add processing time to result
        if isinstance(result, dict) and "metadata" in result:
            result["metadata"]["processing_time"] = processing_time
        
        return result

    def get_stage_status(self, stage_name: str) -> Dict[str, Any]:
        """
        Get status information for a specific processing stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Dictionary with stage status information
        """
        if not self.context or "stages" not in self.context.metadata:
            return {"status": "not_started"}
        
        return self.context.metadata["stages"].get(stage_name, {"status": "not_started"})

    def reset(self) -> None:
        """Reset the orchestrator to process a new document."""
        self.context = None       