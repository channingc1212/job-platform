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
        
        # Get optimization suggestions
        optimization_chain = LLMChain(llm=self.llm, prompt=self.resume_optimization_prompt)
        analysis = optimization_chain.run({
            'resume_text': resume_text,
            'job_requirements': job_requirements
        })
        
        # Generate optimized content
        optimization_content_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["resume_text", "job_requirements", "analysis"],
                template="""Based on the original resume and job requirements, provide an optimized version of the resume content. 
                Maintain the same structure but enhance the content to better match the job requirements.
                
                Original Resume:
                {resume_text}
                
                Job Requirements:
                {job_requirements}
                
                Analysis:
                {analysis}
                
                Instructions:
                1. Keep the same sections and overall format
                2. Enhance bullet points to highlight relevant skills
                3. Adjust wording to match job requirements
                4. Maintain a professional tone
                5. Keep content within one page
                
                Return the optimized resume content in the exact same format as the original."""
            )
        )
        
        optimized_content = optimization_content_chain.run({
            'resume_text': resume_text,
            'job_requirements': job_requirements,
            'analysis': analysis
        })
        
        # Generate a concise summary of changes
        summary_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["original", "optimized"],
                template="""Compare the original and optimized resumes and list the specific changes made:

                Original Resume:
                {original}

                Optimized Resume:
                {optimized}

                Provide a bullet-point list of specific changes made, such as:
                - Added/modified skills
                - Enhanced job descriptions
                - Updated terminology
                - Restructured sections
                
                Be specific and concrete about each change."""
            )
        )
        changes_summary = summary_chain.run({
            'original': resume_text,
            'optimized': optimized_content
        })
        
        # Create optimized PDF with highlighted changes
        optimized_pdf = self._create_optimized_pdf(resume_file, resume_text, optimized_content)
        
        return {
            'analysis': analysis,
            'changes_summary': changes_summary,
            'optimized_resume': optimized_pdf,
            'original_text': resume_text,
            'optimized_text': optimized_content
        }

    def _create_optimized_pdf(self, original_resume, original_text, optimized_text):
        """
        Create an optimized PDF with highlighted changes
        """
        # Create a new PDF
        output = PdfWriter()
        pdf_buffer = BytesIO()
        
        # Create the first page with optimized content
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        
        # Set font and size
        c.setFont("Helvetica", 10)
        y_position = height - inch  # Start 1 inch from top
        
        # Split content into lines
        lines = optimized_text.split('\n')
        
        # Write each line
        for line in lines:
            if y_position < inch:  # Ensure we stay within page bounds
                break
            c.drawString(inch, y_position, line.strip())
            y_position -= 12  # Adjust line spacing
        
        c.save()
        pdf_buffer.seek(0)
        
        # Add the page to output
        optimized_pdf = PdfReader(pdf_buffer)
        output.add_page(optimized_pdf.pages[0])
        
        # Save to a new buffer
        final_buffer = BytesIO()
        output.write(final_buffer)
        final_buffer.seek(0)
        
        return final_buffer

    def _extract_resume_text(self, resume_file):
        """
        Extract text from uploaded resume with improved formatting and structure preservation
        """
        pdf_reader = PdfReader(resume_file)
        text_content = []
        
        for page in pdf_reader.pages:
            # Extract text with better whitespace handling
            text = page.extract_text()
            
            # Split into sections and clean up
            sections = text.split('\n\n')
            cleaned_sections = []
            
            for section in sections:
                # Clean up each section while preserving structure
                lines = section.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    # Remove excessive spaces while preserving indentation
                    cleaned = ' '.join(line.split())
                    if cleaned:  # Only add non-empty lines
                        cleaned_lines.append(cleaned)
                
                if cleaned_lines:  # Only add non-empty sections
                    cleaned_sections.append('\n'.join(cleaned_lines))
            
            # Join sections with proper spacing
            text_content.extend(cleaned_sections)
            text_content.append('')  # Add blank line between pages
        
        # Create final formatted text
        final_text = '\n\n'.join(text_content)
        
        # Clean up any remaining formatting issues
        while '\n\n\n' in final_text:
            final_text = final_text.replace('\n\n\n', '\n\n')
        
        return final_text.strip()
