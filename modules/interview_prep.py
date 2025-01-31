from typing import Dict, List, Optional
import logging
import openai
import json
from bs4 import BeautifulSoup
import requests
from dataclasses import dataclass
from datetime import datetime
import os
import re
from dotenv import load_dotenv

@dataclass
class CompanyReview:
    overall_rating: float
    work_life_balance: float
    compensation: float
    career_growth: float
    culture: float
    pros: List[str]
    cons: List[str]
    additional_metrics: Dict[str, float]
    last_updated: str
    sources: List[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'CompanyReview':
        # Extract known fields
        review_data = {
            'overall_rating': data['overall_rating'],
            'work_life_balance': data['work_life_balance'],
            'compensation': data['compensation'],
            'career_growth': data['career_growth'],
            'culture': data['culture'],
            'pros': data['pros'],
            'cons': data['cons'],
            'last_updated': data['last_updated'],
            'additional_metrics': data.get('additional_metrics', {})
        }
        return cls(**review_data)

@dataclass
class InterviewProcess:
    role: str
    difficulty: float
    duration: str
    stages: List[str]
    common_questions: List[str]
    tips: List[str]
    last_updated: str
    sources: List[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'InterviewProcess':
        return cls(
            role=data['role'],
            difficulty=data['difficulty'],
            duration=data['duration'],
            stages=data['stages'],
            common_questions=data['common_questions'],
            tips=data['tips'],
            last_updated=data['last_updated']
        )

class InterviewPrepManager:
    def __init__(self):
        load_dotenv()
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        
        if not perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
            
        self.perplexity_client = openai.Client(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def _make_api_request(self, messages: List[Dict], temperature: float = 0.7) -> Optional[Dict]:
        """Make a request to the Perplexity API"""
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                "Content-Type": "application/json"
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
                logging.error(f"API error: {response.status_code}")
                logging.error(f"Response: {response.text}")
                return None
                
            return response.json()
        except Exception as e:
            logging.error(f"API request error: {e}")
            return None

    def get_company_info(self, company_url: str) -> Dict:
        """Get company reviews and interview process information"""
        try:
            # First message to get company reviews
            review_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides detailed company reviews and ratings. \n\n"
                              "IMPORTANT: Return ONLY a JSON object with NO additional text or explanation. The JSON must match this EXACT format:\n"
                              "{\n"
                              "  \"overall_rating\": 4.2,\n"
                              "  \"work_life_balance\": 4.0,\n"
                              "  \"compensation\": 4.5,\n"
                              "  \"career_growth\": 4.3,\n"
                              "  \"culture\": 4.1,\n"
                              "  \"pros\": [\"Pro 1\", \"Pro 2\"],\n"
                              "  \"cons\": [\"Con 1\", \"Con 2\"],\n"
                              "  \"additional_metrics\": {\"metric_name\": 4.0},\n"
                              "  \"last_updated\": \"2024-01-31\"\n"
                              "}\n\n"
                              "Focus on these key areas:\n"
                              "1. Overall company rating\n"
                              "2. Work-life balance\n"
                              "3. Compensation and benefits\n"
                              "4. Career growth opportunities\n"
                              "5. Company culture\n"
                              "Include the most frequently mentioned pros and cons from employee reviews."
                },
                {
                    "role": "user",
                    "content": f"Find detailed employee reviews and ratings for the company at this URL: {company_url}\n\n"
                              f"Return the information in this exact JSON format:\n"
                              f"{{\n"
                              f"  \"overall_rating\": 4.2,\n"
                              f"  \"work_life_balance\": 4.0,\n"
                              f"  \"compensation\": 4.5,\n"
                              f"  \"career_growth\": 4.3,\n"
                              f"  \"culture\": 4.1,\n"
                              f"  \"pros\": [\"Pro 1\", \"Pro 2\"],\n"
                              f"  \"cons\": [\"Con 1\", \"Con 2\"],\n"
                              f"  \"additional_metrics\": {{\n"
                              f"    \"metric_name\": rating\n"
                              f"  }},\n"
                              f"  \"last_updated\": \"YYYY-MM-DD\"\n"
                              f"}}"
                }
            ]

            # Second message to get interview process
            interview_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides detailed interview process information for Data Science and Analytics roles ONLY. \n\n"
                              "IMPORTANT: Return ONLY a JSON object with NO additional text or explanation. The JSON must match this EXACT format:\n"
                              "{\n"
                              "  \"role\": \"Data Scientist/Analyst\",\n"
                              "  \"difficulty\": 4.2,\n"
                              "  \"duration\": \"2-3 weeks\",\n"
                              "  \"stages\": [\"Stage 1\", \"Stage 2\"],\n"
                              "  \"common_questions\": [\"Question 1\", \"Question 2\"],\n"
                              "  \"tips\": [\"Tip 1\", \"Tip 2\"],\n"
                              "  \"last_updated\": \"2024-01-31\"\n"
                              "}\n\n"
                              "Focus on:\n"
                              "1. Detailed interview stages\n"
                              "2. Common technical and behavioral questions\n"
                              "3. Typical duration of the process\n"
                              "4. Tips from successful candidates\n"
                              "5. Overall difficulty rating"
                },
                {
                    "role": "user",
                    "content": f"Find the interview process details for Data Science and Analytics roles at this URL: {company_url}\n\n"
                              f"Return the information in this exact JSON format:\n"
                              f"{{\n"
                              f"  \"role\": \"Data Scientist/Analyst\",\n"
                              f"  \"difficulty\": 4.2,\n"
                              f"  \"duration\": \"2-3 weeks\",\n"
                              f"  \"stages\": [\"Stage 1\", \"Stage 2\"],\n"
                              f"  \"common_questions\": [\"Question 1\", \"Question 2\"],\n"
                              f"  \"tips\": [\"Tip 1\", \"Tip 2\"],\n"
                              f"  \"last_updated\": \"YYYY-MM-DD\"\n"
                              f"}}"
                }
            ]

            # Get company reviews
            review_result = self._make_api_request(review_messages)
            if not review_result or 'choices' not in review_result:
                logging.error("Failed to get company reviews")
                return None

            # Get interview process
            interview_result = self._make_api_request(interview_messages)
            if not interview_result or 'choices' not in interview_result:
                logging.error("Failed to get interview process")
                return None

            def extract_json(content: str) -> dict:
                # Try to find JSON in code block
                code_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', content)
                if code_match:
                    json_str = code_match.group(1)
                else:
                    # Try to find raw JSON object
                    json_match = re.search(r'({[\s\S]*?})(?:\s*$|\n)', content)
                    if not json_match:
                        raise ValueError("No JSON object found in content")
                    json_str = json_match.group(1)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON: {e}")
                    logging.error(f"JSON string: {json_str}")
                    raise

            try:
                # Parse review response
                review_content = review_result['choices'][0]['message']['content']
                logging.info(f"Review content: {review_content}")
                
                # Parse review data
                try:
                    review_data = extract_json(review_content)
                    logging.info(f"Parsed review data: {review_data}")
                    # Add sources from citations
                    review_data['sources'] = review_result.get('citations', [])
                    company_review = CompanyReview.from_dict(review_data)
                except Exception as e:
                    logging.error(f"Failed to parse review data: {str(e)}")
                    return None

                # Parse interview response
                interview_content = interview_result['choices'][0]['message']['content']
                logging.info(f"Interview content: {interview_content}")
                
                try:
                    interview_data = extract_json(interview_content)
                    logging.info(f"Parsed interview data: {interview_data}")
                    # Add sources from citations
                    interview_data['sources'] = interview_result.get('citations', [])
                    interview_process = InterviewProcess.from_dict(interview_data)
                except Exception as e:
                    logging.error(f"Failed to parse interview data: {str(e)}")
                    return None

                # Convert to dict and ensure sources are included
                review_dict = company_review.__dict__
                review_dict['sources'] = review_result.get('citations', [])
                
                interview_dict = interview_process.__dict__
                interview_dict['sources'] = interview_result.get('citations', [])
                
                return {
                    "company_review": review_dict,
                    "interview_process": interview_dict
                }

            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}")
                return None
            except AttributeError as e:
                logging.error(f"Missing required fields in response: {e}")
                return None
            except Exception as e:
                logging.error(f"Error processing results: {str(e)}")
                logging.error(f"Review result: {review_result}")
                logging.error(f"Interview result: {interview_result}")
                return None

        except Exception as e:
            logging.error(f"Error getting company info: {e}")
            return None
