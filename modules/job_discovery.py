from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import logging
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Optional
import openai

class JobDiscoveryManager:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
            
        self.perplexity_api_key = perplexity_api_key
        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        
        # Initialize Perplexity client using OpenAI client
        self.perplexity_client = openai.Client(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai")
        
        # Prompt for extracting job preferences from resume
        self.preferences_prompt = PromptTemplate(
            input_variables=["resume_text"],
            template="""
            Extract key job preferences and qualifications from the following resume.
            Focus on:
            1. Technical skills and expertise
            2. Industry experience
            3. Role level (entry, mid, senior)
            4. Previous company types
            5. Educational background
            
            Resume:
            {resume_text}
            
            Return the information in a JSON format with these keys:
            - skills: list of technical skills
            - industries: list of industries worked in
            - role_level: string indicating seniority
            - preferred_companies: list of company types based on previous experience
            - education: dict with 'degree' and 'field'
            """
        )
        
        # Prompt for job search
        self.search_prompt = PromptTemplate(
            input_variables=["preferences", "manual_criteria"],
            template="""
            Find current job openings matching the following preferences and criteria.
            Return results in a structured JSON array format.
            
            Preferences from Resume:
            {preferences}
            
            Additional Criteria:
            {manual_criteria}
            
            Focus on:
            - Job titles matching skills and experience
            - Required skills alignment
            - Company characteristics
            - Location if specified
            """
        )
    
    def extract_preferences(self, resume_file) -> Dict:
        """Extract job preferences from resume"""
        try:
            pdf_reader = PdfReader(resume_file)
            text_content = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                text_content.append(text)
            
            resume_text = "\n".join(text_content)
            
            # Generate preferences using LLM
            chain = LLMChain(llm=self.llm, prompt=self.preferences_prompt)
            preferences_str = chain.run(resume_text)
            
            # Parse the JSON response
            preferences = json.loads(preferences_str)
            return preferences
            
        except Exception as e:
            logging.error(f"Error extracting preferences: {e}")
            return None
    

    def _make_api_request(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
        """Make a request to the Perplexity API using requests directly for better error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            data = {
                "model": "sonar",
                "messages": messages,
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                logging.error(f"Perplexity API error: {response.status_code}")
                logging.error(f"Response content: {response.text}")
                return None
                
            result = response.json()
            return result
            
        except Exception as e:
            logging.error(f"API request error: {e}")
            if hasattr(e, 'response'):
                logging.error(f"Response content: {e.response.text}")
            return None
    
    def search_job_openings(self, *, background: str, criteria: str = "") -> List[Dict]:
        """Search for job openings using Perplexity API
        
        Args:
            background: Candidate's background and preferences (from resume)
            criteria: Additional search criteria (manual input)
            
        Returns:
            List of job openings matching both background and criteria
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that finds current job openings based on candidate background and requirements. Focus on the following criteria:\n\n" 
                               "1. Job Functions: Analytics or Data Science roles\n" 
                               "2. Industry: Technology\n" 
                               "3. Company Stage: Series B or later\n" 
                               "4. Location: San Francisco Bay Area or Remote\n" 
                               "5. Company Type: US-based companies only\n\n" 
                               "Return ONLY a JSON array containing relevant job listings that match BOTH:\n" 
                               "- The candidate's background and requirements\n" 
                               "- The default criteria listed above\n\n" 
                               "Ensure all returned jobs are currently open positions."
                },
                {
                    "role": "user",
                    "content": f"Find current job openings matching the following candidate profile and requirements:\n\n"
                               f"Candidate Background & Preferences:\n{background}\n\n"
                               f"Additional Requirements:\n{criteria if criteria else 'None specified'}\n\n"
                               f"Additional Instructions:\n"
                               f"1. Return response in this exact JSON format:\n"
                               f"[{{"
                               f"\n  \"title\": \"Job Title\","
                               f"\n  \"company\": \"Company Name\","
                               f"\n  \"location\": \"Job Location\","
                               f"\n  \"description\": \"Brief job description\","
                               f"\n  \"requirements\": [\"Requirement 1\", \"Requirement 2\"],"
                               f"\n  \"link\": \"Application URL\","
                               f"\n  \"posted_date\": \"Posting Date\","
                               f"\n  \"salary\": \"Salary range if available\""
                               f"\n}}]\n"
                               f"2. Focus on currently open positions\n"
                               f"3. Ensure all jobs match the candidate's background and requirements\n"
                }
            ]
            
            logging.info(f"Sending job search request with background: {background} and criteria: {criteria}")
            result = self._make_api_request(messages)
            logging.info(f"Got API response: {result}")
            if not result or 'choices' not in result:
                logging.error(f"Invalid API response format: {result}")
                return []
            
            try:
                content = result['choices'][0]['message']['content']
                # Try to parse the entire response as JSON first
                try:
                    jobs = json.loads(content)
                except json.JSONDecodeError:
                    # If that fails, try to find a JSON array in the content
                    import re
                    json_match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
                    if json_match:
                        try:
                            jobs = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            logging.error(f"Failed to parse JSON array: {json_match.group()}")
                            return []
                    else:
                        logging.error(f"No JSON array found in content: {content}")
                        return []
                
                if not isinstance(jobs, list):
                    jobs = [jobs] if isinstance(jobs, dict) else []
                
                # Validate each job has required fields
                required_fields = ['title', 'company', 'location', 'description', 'requirements']
                jobs = [job for job in jobs if all(field in job for field in required_fields)]
                
                return jobs
                
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing job listings JSON: {e}")
                logging.error(f"Raw content: {content}")
                return []
                
        except Exception as e:
            logging.error(f"Error searching jobs: {e}")
            return []
    
    def get_company_info(self, company_name: str) -> Dict:
        """Get detailed company information"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides company information. Return results in JSON format."
                },
                {
                    "role": "user",
                    "content": f"Find company information for {company_name}. Include: founding year, size, funding, financial performance, headquarters. Return as JSON."
                }
            ]
            
            result = self._make_api_request(messages)
            if not result:
                return {}
            
            content = result['choices'][0]['message']['content']
            try:
                company_info = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON-like content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        company_info = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        company_info = {}
                else:
                    company_info = {}
            return company_info
                
        except Exception as e:
            logging.error(f"Error getting company info: {e}")
            return {}
