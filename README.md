# Job Hunt Assistant

A powerful tool to streamline your job hunting process using AI. This application helps you optimize your resume and create personalized outreach messages.

## Features

### 1. Resume Optimization
- Upload your resume (PDF or DOCX format)
- Get detailed analysis of your resume against job descriptions
- Receive specific recommendations for improvements
- Identify skill gaps and areas to highlight

### 2. Outreach Message Generator
- Generate personalized outreach messages for LinkedIn or email
- Customize messages based on company, role, and your background
- Create engaging and professional communication

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd job-platform
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the root directory
- Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```
Note: The `.env` file is automatically ignored by git for security.

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Choose a feature from the sidebar:
   - Resume Optimization
   - Outreach Message Generator

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
