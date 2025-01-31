from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import logging
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

class OutreachManager:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        
        # Prompt for extracting background from resume
        self.background_prompt = PromptTemplate(
            input_variables=["resume_text"],
            template="""
            Extract a concise professional background summary from the following resume.
            Focus on key achievements, skills, and experiences that would be relevant for job applications.
            Keep it under 3-4 sentences.
            
            Resume:
            {resume_text}
            """
        )
        
        # Prompt for generating outreach message
        self.outreach_prompt = PromptTemplate(
            input_variables=["company_name", "role", "background", "interests"],
            template="""
            Write a professional and personalized outreach message for a job application with the following details:
            
            Company: {company_name}
            Role: {role}
            Your Background: {background}
            Why Interested: {interests}
            
            The message should:
            1. Be concise but engaging
            2. Show genuine interest in the company and role
            3. Highlight relevant experience
            4. Include a clear call to action
            5. Maintain a professional yet conversational tone
            6. Avoid the tone of a formal or AI-generated message
            
            Write the message in a format suitable for LinkedIn or email:
            """
        )
        
        self.background_chain = LLMChain(llm=self.llm, prompt=self.background_prompt)
        self.outreach_chain = LLMChain(llm=self.llm, prompt=self.outreach_prompt)
    
    def extract_job_info(self, job_url: str) -> dict:
        """
        Extract job information from the provided URL
        """
        try:
            response = requests.get(job_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple common job description selectors
            job_desc_selectors = [
                'div.job-description', 
                'div#job-description', 
                'div.description', 
                'div.jobDescriptionContent',
                'body'
            ]
            
            job_description = None
            for selector in job_desc_selectors:
                element = soup.select_one(selector)
                if element:
                    job_description = element.get_text(strip=True)
                    break
            
            # Extract company name from common locations
            company_selectors = [
                'div.company-name',
                'a.company-name',
                'span.company',
                'div.company'
            ]
            
            company_name = None
            for selector in company_selectors:
                element = soup.select_one(selector)
                if element:
                    company_name = element.get_text(strip=True)
                    break
            
            return {
                'company_name': company_name,
                'job_description': job_description
            }
        except Exception as e:
            logging.error(f"Error extracting job info: {e}")
            return None
    
    def extract_background(self, resume_file) -> str:
        """
        Extract and summarize background information from resume
        """
        try:
            pdf_reader = PdfReader(resume_file)
            text_content = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                text_content.append(text)
            
            resume_text = "\n".join(text_content)
            background = self.background_chain.run(resume_text)
            return background
        except Exception as e:
            logging.error(f"Error extracting background: {e}")
            return None

    def generate_message(self, resume_file, job_url: str, specific_interests: str = "") -> str:
        """
        Generate a personalized outreach message based on resume and job URL
        """
        try:
            # Extract background from resume
            background = self.extract_background(resume_file)
            if not background:
                return "Error: Could not extract background from resume"
            
            # Extract job info from URL
            job_info = self.extract_job_info(job_url)
            if not job_info:
                return "Error: Could not extract job information from URL"
            
            # Generate message
            message = self.outreach_chain.run({
                "company_name": job_info['company_name'] or "the company",
                "role": job_info['job_description'] or "the role",
                "background": background,
                "interests": specific_interests
            })
            return message
        except Exception as e:
            return f"Error generating message: {str(e)}"
