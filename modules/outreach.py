from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import logging

class OutreachManager:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Configure for project-specific API key
        os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key,
            openai_api_base="https://api.openai.com/v1"  # Explicitly set API base
        )
        self.prompt = PromptTemplate(
            input_variables=["company_name", "contact_name", "role", "background", "interests"],
            template="""
            Write a professional and personalized outreach message for a job application with the following details:
            
            Company: {company_name}
            Contact: {contact_name}
            Role: {role}
            Your Background: {background}
            Specific Interests: {interests}
            
            The message should:
            1. Be concise but engaging
            2. Show genuine interest in the company and role
            3. Highlight relevant experience
            4. Include a clear call to action
            5. Maintain a professional yet conversational tone
            
            Write the message in a format suitable for LinkedIn or email:
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_message(self, company_name: str, contact_name: str, role: str, 
                        background: str, interests: str) -> str:
        """
        Generate a personalized outreach message based on the provided information.
        """
        try:
            message = self.chain.run({
                "company_name": company_name,
                "contact_name": contact_name if contact_name else "[Hiring Manager]",
                "role": role,
                "background": background,
                "interests": interests
            })
            return message
        except Exception as e:
            return f"Error generating message: {str(e)}"
