from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from PyPDF2 import PdfReader, PdfWriter
from docx import Document
import io
import os
import logging
import requests
from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO
from dotenv import load_dotenv

class ResumeOptimizer:
    def __init__(self):
        # Load environment variables
        load_dotenv(override=True)
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize LLM with explicit API key
        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key  # Explicitly pass the API key
        )

        self.job_description_prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""Extract the key requirements and skills from the following job description:
            
            {job_description}
            
            Provide a concise, bulleted list of the most important skills and qualifications."""
        )

        self.resume_optimization_prompt = PromptTemplate(
            input_variables=["resume_text", "job_requirements"],
            template="""Analyze the resume in the context of the job requirements:

            Resume:
            {resume_text}

            Job Requirements:
            {job_requirements}

            Provide a detailed analysis of:
            1. How well the resume matches the job requirements
            2. Specific areas for improvement
            3. Recommended changes to increase the resume's effectiveness"""
        )

    def extract_job_description(self, job_url):
        """
        Extract job description from a given URL
        """
        try:
            response = requests.get(job_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple common job description selectors
            job_desc_selectors = [
                'div.job-description', 
                'div#job-description', 
                'div.description', 
                'div.jobDescriptionContent',
                'body'  # Fallback to entire body if no specific selector found
            ]
            
            for selector in job_desc_selectors:
                job_desc_element = soup.select_one(selector)
                if job_desc_element:
                    return job_desc_element.get_text(strip=True)
            
            return None
        except Exception as e:
            logging.error(f"Error extracting job description: {e}")
            return None
    
    def analyze_resume(self, resume_file, job_description=None, job_url=None):
        """
        Analyze and potentially optimize resume
        """
        # Extract job description from URL if provided
        if job_url and not job_description:
            job_description = self.extract_job_description(job_url)
        
        if not job_description:
            return "No job description provided or could not be extracted."
        
        # Extract job requirements
        job_requirements_chain = LLMChain(llm=self.llm, prompt=self.job_description_prompt)
        job_requirements = job_requirements_chain.run(job_description)
        
        # Extract resume text
        resume_text = self._extract_resume_text(resume_file)
        
        # Analyze resume optimization
        optimization_chain = LLMChain(llm=self.llm, prompt=self.resume_optimization_prompt)
        optimization_analysis = optimization_chain.run({
            'resume_text': resume_text,
            'job_requirements': job_requirements
        })
        
        # Create optimized PDF (single page)
        optimized_pdf = self._create_optimized_pdf(resume_file)
        
        # Summarize key changes
        changes_summary = self._summarize_changes(optimization_analysis)
        
        return {
            'analysis': optimization_analysis,
            'changes_summary': changes_summary,
            'optimized_resume': optimized_pdf
        }
    
    def _summarize_changes(self, optimization_analysis):
        """
        Create a concise summary of recommended changes
        """
        try:
            summary_chain = LLMChain(
                llm=self.llm, 
                prompt=PromptTemplate(
                    input_variables=["optimization_analysis"],
                    template="""
                    Provide a concise, bullet-point summary of the key recommended changes 
                    from the resume optimization analysis. Focus on the most impactful 
                    suggestions that would improve the resume's alignment with the job requirements.
                    
                    Optimization Analysis:
                    {optimization_analysis}
                    
                    Summary should be:
                    - No more than 3-5 key points
                    - Actionable and specific
                    - Highlight the most critical improvements
                    """
                )
            )
            
            summary = summary_chain.run(optimization_analysis)
            return summary
        except Exception as e:
            logging.error(f"Error generating changes summary: {e}")
            return "No significant changes recommended."
    
    def _create_optimized_pdf(self, original_resume):
        """
        Create an optimized PDF that is exactly the same as the original
        Ensures only one page is preserved
        """
        # Read original PDF
        original_pdf = PdfReader(original_resume)
        original_page = original_pdf.pages[0]
        
        # Create a new PDF with only the first page
        output = PdfWriter()
        output.add_page(original_page)
        
        # Save to a BytesIO buffer
        pdf_buffer = BytesIO()
        output.write(pdf_buffer)
        pdf_buffer.seek(0)
        
        return pdf_buffer
    
    def _extract_resume_text(self, resume_file):
        """
        Extract text from uploaded resume
        """
        pdf_reader = PdfReader(resume_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
