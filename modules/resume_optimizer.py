from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from docx import Document
import io

class ResumeOptimizer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        self.prompt = PromptTemplate(
            input_variables=["resume_text", "job_description"],
            template="""
            Analyze the resume and job description provided below and provide detailed feedback:

            Resume:
            {resume_text}

            Job Description:
            {job_description}

            Please provide analysis in the following format:

            1. Skills Match:
            - List matching skills between resume and job requirements
            - Identify missing or underrepresented skills
            
            2. Experience Alignment:
            - How well does the experience align with the role?
            - Key achievements that are relevant
            - Areas where experience could be better highlighted
            
            3. Specific Recommendations:
            - Bullet points of suggested changes or additions
            - Keywords to incorporate
            - Achievements to emphasize
            
            4. Overall Assessment:
            - Brief summary of fit for the role
            - Top 3 suggested improvements
            
            Provide specific, actionable feedback that will help improve the resume for this particular role.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _extract_text_from_pdf(self, file) -> str:
        """Extract text from a PDF file."""
        try:
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"
    
    def _extract_text_from_docx(self, file) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = Document(file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            return f"Error extracting DOCX text: {str(e)}"
    
    def analyze_resume(self, resume_file, job_description: str) -> str:
        """
        Analyze the resume against the job description and provide recommendations.
        """
        try:
            # Extract text from resume file
            file_extension = resume_file.name.split('.')[-1].lower()
            if file_extension == 'pdf':
                resume_text = self._extract_text_from_pdf(resume_file)
            elif file_extension == 'docx':
                resume_text = self._extract_text_from_docx(resume_file)
            else:
                return "Unsupported file format. Please upload a PDF or DOCX file."
            
            # Generate analysis
            analysis = self.chain.run({
                "resume_text": resume_text,
                "job_description": job_description
            })
            return analysis
        except Exception as e:
            return f"Error analyzing resume: {str(e)}"
