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
from langsmith import Client
from langsmith.run_helpers import traceable
import datetime
import time
import uuid
import traceback
import functools

def get_langsmith_project():
    """Get the LangSmith project name from environment variables"""
    return os.getenv("LANGSMITH_PROJECT", "job-platform")

def create_trace(run_type: str, name: str):
    """Create a traceable decorator with proper project configuration"""
    return traceable(
        run_type=run_type,
        name=name,
        project_name=get_langsmith_project()
    )

class JobDiscoveryManager:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        
        # Initialize LangSmith client if API key is available
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "job-platform")
        
        # Ensure tracing is enabled
        os.environ["LANGSMITH_TRACING"] = "true"
        
        if self.langsmith_api_key:
            try:
                self.langsmith_client = Client(
                    api_key=self.langsmith_api_key
                )
                # Try to create project (will use existing if it already exists)
                self.langsmith_client.create_project(
                    project_name=self.langsmith_project,
                    description="Job discovery platform traces"
                )
                logging.info(f"LangSmith initialized with project: {self.langsmith_project}")
                
                # Enable debug logging for LangSmith
                logging.getLogger('langsmith').setLevel(logging.DEBUG)
                
                # Log environment state
                logging.debug(f"LangSmith Environment: TRACING={os.getenv('LANGSMITH_TRACING')}, PROJECT={self.langsmith_project}")
            except Exception as e:
                logging.error(f"Error initializing LangSmith: {e}")
                self.langsmith_client = None
        else:
            self.langsmith_client = None
            logging.warning("LANGSMITH_API_KEY not found. LangSmith tracing disabled.")

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
        
        # Load search configurations
        self.search_configs = self._load_search_configs()
        self.active_config_name = os.getenv("ACTIVE_SEARCH_CONFIG", "default")
        logging.info(f"Using search configuration: {self.active_config_name}")
        
        # Prompt templates
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
        
    def _load_search_configs(self) -> Dict:
        """Load search configuration templates from config file or create default if not exists"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'search_configs.json')
        
        # Create default config dictionary
        default_configs = {
            "default": {
                "description": "Default search configuration",
                "system_prompt": """You are a helpful assistant that finds current job openings based on candidate background and requirements. 
                
                Focus on the following criteria:
                1. Job Functions: Analytics or Data Science roles
                2. Industry: Technology
                3. Company Stage: Series B or later
                4. Location: San Francisco Bay Area or Remote
                5. Company Type: US-based companies only
                
                Return ONLY a JSON array containing relevant job listings that match BOTH:
                - The candidate's background and requirements
                - The default criteria listed above
                
                Ensure all returned jobs are currently open positions and each job is a separate, complete entry.
                Each job MUST have a specific company name and application link.""",
                "user_prompt_template": """Find current job openings matching the following candidate profile and requirements:

Candidate Background & Preferences:
{background}

Additional Requirements:
{criteria}

Additional Instructions:
1. Return response in this exact JSON format:
[{{
  "title": "Job Title",
  "company": "Company Name (must be a specific, real company)",
  "location": "Job Location",
  "description": "Brief job description",
  "requirements": ["Requirement 1", "Requirement 2"],
  "link": "Application URL (use a real job board URL)",
  "posted_date": "Recent date",
  "salary": "Salary range if available"
}}]
2. Focus on currently open positions
3. Ensure all jobs match the candidate's background and requirements
4. Make sure each job is a separate, complete entry in the array
5. IMPORTANT: Ensure every job has a specific company name (not N/A or Unknown)
6. IMPORTANT: Ensure every job has a valid application link
7. If you find multiple positions at the same company, create separate entries for each position
8. If no specific jobs match the criteria, provide at least 1-2 relevant job suggestions that are close matches""",
                "model": "sonar",
                "temperature": 0.5,
                "use_fallback": True,
                "metrics": {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "average_jobs_returned": 0,
                    "average_response_time": 0
                },
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat()
            },
            "specific_jobs": {
                "description": "More targeted job search with specific job board focus",
                "system_prompt": """You are a helpful assistant that finds current job openings based on candidate background and requirements.
                
                When searching for jobs, prioritize these sources:
                1. LinkedIn Jobs
                2. Indeed
                3. Glassdoor
                4. Company career pages
                
                Focus on finding SPECIFIC, REAL job postings that are currently open, rather than general recommendations.
                Each job must include a real company name, position title, and application link.""",
                "user_prompt_template": """Find specific, real job openings matching the following candidate profile:

Candidate Background & Preferences:
{background}

Additional Requirements:
{criteria}

Instructions:
1. Return ONLY real job postings that are currently open
2. Include specific details for each job (title, company, location, etc.)
3. Ensure each job has a valid application link to the original posting
4. Format the response as a JSON array of job objects with these fields:
   - title: Job title
   - company: Company name
   - location: Job location
   - description: Brief job description
   - requirements: List of key requirements
   - link: Direct application URL
   - posted_date: When the job was posted
   - salary: Salary information if available
   - source: Which job board or website this was found on (LinkedIn, Indeed, etc.)""",
                "model": "sonar",
                "temperature": 0.3,
                "use_fallback": True,
                "metrics": {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "average_jobs_returned": 0,
                    "average_response_time": 0
                },
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat()
            }
        }
        
        # Create config directory if it doesn't exist
        config_dir = os.path.dirname(config_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # Load config file if exists, otherwise create default
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    configs = json.load(f)
                logging.info(f"Loaded {len(configs)} search configurations from {config_path}")
                return configs
            except Exception as e:
                logging.error(f"Error loading search configs: {e}, using defaults")
                return default_configs
        else:
            # Write default configs to file
            try:
                with open(config_path, 'w') as f:
                    json.dump(default_configs, f, indent=2)
                logging.info(f"Created default search configurations at {config_path}")
            except Exception as e:
                logging.error(f"Error creating default search configs: {e}")
            
            return default_configs
    
    def save_search_metrics(self, config_name: str, metrics_update: Dict):
        """Update metrics for a search configuration"""
        if config_name in self.search_configs:
            self.search_configs[config_name]["metrics"].update(metrics_update)
            self.search_configs[config_name]["updated_at"] = datetime.datetime.now().isoformat()
            
            # Save updated configs to file
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'search_configs.json')
            try:
                with open(config_path, 'w') as f:
                    json.dump(self.search_configs, f, indent=2)
                logging.info(f"Updated search metrics for configuration: {config_name}")
            except Exception as e:
                logging.error(f"Error saving search metrics: {e}")
    
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
    
    @create_trace(run_type="llm", name="perplexity_api_request")
    def _make_api_request(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
    # Method implementation
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
            
            logging.info(f"API Request - Model: sonar, Temperature: {temperature}")
            logging.info(f"API Request - System Message: {messages[0]['content'] if messages and len(messages) > 0 else 'None'}")
            logging.info(f"API Request - User Message: {messages[1]['content'] if messages and len(messages) > 1 else 'None'}")
            
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
            
            if result and 'choices' in result and len(result['choices']) > 0:
                content_sample = result['choices'][0]['message']['content'][:500] + "..." if len(result['choices'][0]['message']['content']) > 500 else result['choices'][0]['message']['content']
                logging.info(f"API Response (sample): {content_sample}")
            
            return result
            
        except Exception as e:
            logging.error(f"API request error: {e}")
            if hasattr(e, 'response'):
                logging.error(f"Response content: {e.response.text}")
            return None

    @create_trace(run_type="chain", name="job_search")
    def search_job_openings(self, *, background: str, criteria: str = "", config_name: str = None) -> List[Dict]:
        """Search for job openings using Perplexity API
        
        Args:
            background: Candidate's background and preferences (from resume)
            criteria: Additional search criteria (manual input)
            config_name: Name of the search configuration to use (defaults to active_config_name)
            
        Returns:
            List of job openings matching both background and criteria
        """
        # Use the specified config or fall back to the active config
        config_name = config_name or self.active_config_name
        
        if config_name not in self.search_configs:
            logging.warning(f"Configuration '{config_name}' not found, using default")
            config_name = "default"
        
        config = self.search_configs[config_name]
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Log the search configuration being used
            logging.info(f"Job Search [{request_id}] - Using configuration: {config_name}")
            logging.info(f"Job Search [{request_id}] - Background length: {len(background)} chars, Criteria: {criteria}")
            
            # Format the user prompt with background and criteria
            user_prompt = config["user_prompt_template"].format(
                background=background,
                criteria=criteria if criteria else "None specified"
            )
            
            messages = [
                {
                    "role": "system",
                    "content": config["system_prompt"]
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            logging.info(f"Job Search [{request_id}] - Sending request with model: {config['model']}, temperature: {config['temperature']}")
            
            result = self._make_api_request(messages, temperature=config['temperature'])
            
            # Update metrics for total runs
            current_metrics = config["metrics"]
            current_metrics["total_runs"] += 1
            
            if not result or 'choices' not in result:
                logging.error(f"Job Search [{request_id}] - Invalid API response format")
                self.save_search_metrics(config_name, current_metrics)
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
                            logging.error(f"Job Search [{request_id}] - Failed to parse JSON array")
                            return []
                    else:
                        logging.error(f"Job Search [{request_id}] - No JSON array found in content")
                        return []
                
                if not isinstance(jobs, list):
                    jobs = [jobs] if isinstance(jobs, dict) else []
                
                # Validate and normalize each job
                normalized_jobs = []
                for job in jobs:
                    # Check for required fields
                    required_fields = ['title', 'company', 'location', 'description', 'requirements', 'link']
                    
                    # Skip jobs with missing required fields or with placeholder values
                    if not all(field in job for field in required_fields):
                        continue
                        
                    # Skip jobs with placeholder company names
                    if job.get('company', '').lower() in ['n/a', 'unknown', 'unknown company', '']:
                        continue
                        
                    # Add search metadata
                    job['_metadata'] = {
                        'search_id': request_id,
                        'config_name': config_name,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    
                    # Normalize the job data
                    normalized_job = {
                        'title': job.get('title', 'Unknown Title').strip(),
                        'company': job.get('company', 'Unknown Company').strip(),
                        'location': job.get('location', 'Unknown Location').strip(),
                        'description': job.get('description', 'No description available').strip(),
                        'requirements': job.get('requirements', []),
                        'link': job.get('link', 'https://www.linkedin.com/jobs/').strip(),
                        'posted_date': job.get('posted_date', 'Unknown').strip(),
                        'salary': job.get('salary', 'Not specified').strip(),
                        '_metadata': job.get('_metadata', {})
                    }
                    
                    # Ensure requirements is a list
                    if not isinstance(normalized_job['requirements'], list):
                        if isinstance(normalized_job['requirements'], str):
                            normalized_job['requirements'] = [normalized_job['requirements']]
                        else:
                            normalized_job['requirements'] = []
                    
                    # Ensure link is a valid URL
                    if not normalized_job['link'].startswith(('http://', 'https://')):
                        normalized_job['link'] = f"https://{normalized_job['link']}"
                    
                    # Add to normalized jobs if it passes all checks
                    normalized_jobs.append(normalized_job)
                
                # Calculate and update metrics
                duration = time.time() - start_time
                
                # Update metrics with this run's data
                successful = len(normalized_jobs) > 0
                metrics_update = {
                    "successful_runs": current_metrics["successful_runs"] + (1 if successful else 0),
                    "average_jobs_returned": (
                        (current_metrics["average_jobs_returned"] * current_metrics["total_runs"] + len(normalized_jobs)) / 
                        (current_metrics["total_runs"] + 1)
                    ),
                    "average_response_time": (
                        (current_metrics["average_response_time"] * current_metrics["total_runs"] + duration) / 
                        (current_metrics["total_runs"] + 1)
                    )
                }
                
                self.save_search_metrics(config_name, metrics_update)
                
                # Log results
                logging.info(f"Job Search [{request_id}] - Found {len(normalized_jobs)} valid jobs, duration: {duration:.2f}s")
                
                # If no valid jobs were found and fallback is enabled, try the fallback search
                if not normalized_jobs and config.get("use_fallback", True):
                    logging.info(f"Job Search [{request_id}] - No valid jobs found, attempting fallback search")
                    return self._fallback_job_search(background, criteria, request_id)
                
                return normalized_jobs
                
            except json.JSONDecodeError as e:
                logging.error(f"Job Search [{request_id}] - Error parsing job listings JSON: {e}")
                return []
                
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"Job Search [{request_id}] - Error after {duration:.2f}s: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _fallback_job_search(self, background: str, criteria: str = "", request_id: str = "") -> List[Dict]:
        """Fallback method for job search when the primary search returns no results
        
        Args:
            background: Candidate's background and preferences
            criteria: Additional search criteria
            
        Returns:
            List of job openings that are more general matches
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that finds job openings. Your task is to provide at least 2-3 
                    realistic job listings that match the candidate's background, even if they're not perfect matches.
                    
                    Focus on providing REAL companies and REALISTIC job postings with specific details."""
                },
                {
                    "role": "user",
                    "content": f"""The initial job search returned no results. Please provide at least 2-3 realistic job listings 
                    that might be of interest to a candidate with the following background:
                    
                    {background}
                    
                    Additional criteria: {criteria if criteria else 'None specified'}
                    
                    Return in this JSON format:
                    [{{
                      "title": "Job Title",
                      "company": "Company Name (must be a specific, real company)",
                      "location": "Job Location",
                      "description": "Brief job description",
                      "requirements": ["Requirement 1", "Requirement 2"],
                      "link": "Application URL (use a real job board URL)",
                      "posted_date": "Recent date",
                      "salary": "Salary range if available"
                    }}]
                    
                    IMPORTANT: 
                    1. Each job MUST have a specific company name (not N/A or Unknown)
                    2. Each job MUST have a valid application link
                    3. Provide realistic, specific details for each job"""
                }
            ]
            
            result = self._make_api_request(messages, temperature=0.7)  # Higher temperature for more creative results
            if not result or 'choices' not in result:
                return []
            
            content = result['choices'][0]['message']['content']
            try:
                jobs = json.loads(content)
                
                if not isinstance(jobs, list):
                    jobs = [jobs] if isinstance(jobs, dict) else []
                
                # Normalize the fallback jobs
                normalized_jobs = []
                for job in jobs:
                    if 'company' in job and 'title' in job:
                        normalized_job = {
                            'title': job.get('title', 'Unknown Title').strip(),
                            'company': job.get('company', 'Unknown Company').strip(),
                            'location': job.get('location', 'Unknown Location').strip(),
                            'description': job.get('description', 'No description available').strip(),
                            'requirements': job.get('requirements', []),
                            'link': job.get('link', 'https://www.linkedin.com/jobs/').strip(),
                            'posted_date': job.get('posted_date', 'Unknown').strip(),
                            'salary': job.get('salary', 'Not specified').strip()
                        }
                        
                        # Ensure requirements is a list
                        if not isinstance(normalized_job['requirements'], list):
                            if isinstance(normalized_job['requirements'], str):
                                normalized_job['requirements'] = [normalized_job['requirements']]
                            else:
                                normalized_job['requirements'] = []
                        
                        # Ensure link is a valid URL
                        if not normalized_job['link'].startswith(('http://', 'https://')):
                            normalized_job['link'] = f"https://{normalized_job['link']}"
                        
                        normalized_jobs.append(normalized_job)
                
                return normalized_jobs
                
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            logging.error(f"Error in fallback job search: {e}")
            return []
    
    def get_company_info(self, company_name: str) -> Dict:
        """Get detailed company information
        
        Args:
            company_name: Name of the company to get information for
            
        Returns:
            Dictionary containing structured company information or a list of company information
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides company information. Return results in a clean, structured JSON format."
                },
                {
                    "role": "user",
                    "content": f"""Find company information for {company_name}. 
                    
                    If this appears to be a single company, return a JSON object with these fields:
                    - name: Company name
                    - founding_year: Year founded
                    - size: Company size (employees)
                    - funding: Funding information
                    - financial_performance: Financial performance information
                    - headquarters: Headquarters location
                    
                    If this appears to be multiple companies or contains multiple company information, return a JSON array of objects, each with the fields above.
                    
                    Make sure to properly separate and structure the data for each individual company."""
                }
            ]
            
            result = self._make_api_request(messages)
            if not result:
                return {}
            
            content = result['choices'][0]['message']['content']
            
            # Try multiple parsing approaches
            return self._parse_company_response(content)
                
        except Exception as e:
            logging.error(f"Error getting company info: {e}")
            return {}
    
    def _parse_company_response(self, content: str) -> Dict:
        """Parse company information response with multiple fallback strategies
        
        Args:
            content: Raw response content from API
            
        Returns:
            Structured company information
        """
        # First try to parse the entire content as JSON
        try:
            parsed_data = json.loads(content)
            return self._normalize_company_data(parsed_data)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON-like content with regex
        import re
        
        # Try to find an array of companies
        array_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if array_match:
            try:
                parsed_data = json.loads(array_match.group())
                return self._normalize_company_data(parsed_data)
            except json.JSONDecodeError:
                pass
        
        # Try to find a single company object
        obj_match = re.search(r'\{[^{]*\}', content, re.DOTALL)
        if obj_match:
            try:
                parsed_data = json.loads(obj_match.group())
                return self._normalize_company_data(parsed_data)
            except json.JSONDecodeError:
                pass
        
        # If all parsing attempts fail, try to extract structured data with a more lenient approach
        try:
            # Use OpenAI to parse the unstructured text into structured data
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that parses unstructured company information into structured JSON."
                },
                {
                    "role": "user",
                    "content": f"""Parse the following company information into a clean JSON format:
                    
                    {content}
                    
                    If this contains information about multiple companies, return a JSON array of objects.
                    If it's a single company, return a JSON object.
                    
                    Each company object should have these fields (where available):
                    - name: Company name
                    - founding_year: Year founded
                    - size: Company size (employees)
                    - funding: Funding information
                    - financial_performance: Financial performance information
                    - headquarters: Headquarters location
                    """
                }
            ]
            
            parse_result = self._make_api_request(messages)
            if parse_result and 'choices' in parse_result:
                parsed_content = parse_result['choices'][0]['message']['content']
                try:
                    parsed_data = json.loads(parsed_content)
                    return self._normalize_company_data(parsed_data)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logging.error(f"Error in fallback parsing: {e}")
        
        # Return empty dict if all parsing attempts fail
        return {}
    
    def _normalize_company_data(self, data):
        """Normalize company data to ensure consistent structure
        
        Args:
            data: Parsed company data (dict or list)
            
        Returns:
            Normalized company data
        """
        # If data is a list, process each item
        if isinstance(data, list):
            companies = []
            for company in data:
                if isinstance(company, dict):
                    # Ensure each company has all expected fields
                    normalized = {
                        "name": company.get("name", "Unknown"),
                        "founding_year": company.get("founding_year", "N/A"),
                        "size": company.get("size", "N/A"),
                        "funding": company.get("funding", "N/A"),
                        "financial_performance": company.get("financial_performance", "N/A"),
                        "headquarters": company.get("headquarters", "N/A")
                    }
                    companies.append(normalized)
            return {"companies": companies}
        
        # If data is a dict, normalize it
        elif isinstance(data, dict):
            return {
                "name": data.get("name", "Unknown"),
                "founding_year": data.get("founding_year", "N/A"),
                "size": data.get("size", "N/A"),
                "funding": data.get("funding", "N/A"),
                "financial_performance": data.get("financial_performance", "N/A"),
                "headquarters": data.get("headquarters", "N/A")
            }
        
        return {}
