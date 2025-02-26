import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from modules.resume_optimizer import ResumeOptimizer
from modules.outreach import OutreachManager
from modules.job_discovery import JobDiscoveryManager
from modules.interview_prep import InterviewPrepManager
import json
import logging
import pandas as pd
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure LangSmith logging
langsmith_logger = logging.getLogger('langsmith')
langsmith_logger.setLevel(logging.DEBUG)

# Add a stream handler if none exists
if not langsmith_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    langsmith_logger.addHandler(handler)

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Log environment state for debugging
logging.debug(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
logging.debug(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")

# Verify OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("""
    OpenAI API key not found. Please ensure:
    1. Your .env file exists in the project root
    2. It contains: OPENAI_API_KEY=your-key-here
    3. The key is valid and not expired
    """)
    st.stop()

st.set_page_config(page_title="Job Hunt Assistant", layout="wide")
st.title("Your Job Hunt Assistant")

# Create tabs for navigation
resume_tab, outreach_tab, discovery_tab, interview_tab, debug_tab = st.tabs([
    "📝 Resume Optimization", 
    "✉️ Outreach", 
    "🔍 Job Discovery", 
    "🎯 Interview Prep",
    "🔧 Debug"
])

with resume_tab:
    st.header("Resume Optimization")
    
    # Job Description Input
    job_input_method = st.radio(
        "How would you like to provide the job description?", 
        ["Text Input", "Job Posting URL"]
    )
    
    job_description = None
    job_url = None
    
    if job_input_method == "Text Input":
        job_description = st.text_area("Paste Job Description")
    else:
        job_url = st.text_input("Enter Job Posting URL")
    
    # Resume Upload
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    # Optimize Button
    if st.button("Optimize Resume"):
        if resume_file is not None and (job_description or job_url):
            try:
                with st.spinner("Analyzing and optimizing your resume..."):
                    # Perform resume optimization
                    optimizer = ResumeOptimizer()
                    result = optimizer.analyze_resume(
                        resume_file, 
                        job_description=job_description, 
                        job_url=job_url
                    )

                    # Store result in session state
                    st.session_state.optimization_result = result
                
                # Display Optimization Results
                st.success("✅ Resume optimization completed!")
                
                # Display recommended changes prominently
                st.subheader("📝 Changes Made to Your Resume")
                st.info(result['changes_summary'])
                
                # Create tabs for different views
                original_tab, optimized_tab, analysis_tab = st.tabs([
                    "Original Resume", 
                    "Optimized Resume", 
                    "Detailed Analysis"
                ])
                
                with original_tab:
                    st.markdown("### Original Content")
                    st.text_area(
                        "",
                        st.session_state.optimization_result['original_text'],
                        height=400,
                        disabled=True,
                        key="original_text",
                        label_visibility="collapsed",
                        max_chars=None
                    )
                
                with optimized_tab:
                    st.markdown("### Optimized Content")
                    st.text_area(
                        "",
                        st.session_state.optimization_result['optimized_text'],
                        height=400,
                        disabled=True,
                        key="optimized_text",
                        label_visibility="collapsed",
                        max_chars=None
                    )
                    
                    # Store optimization result in session state
                    if 'optimization_result' not in st.session_state:
                        st.session_state['optimization_result'] = result
                    
                    # Download button in the optimized tab
                    st.download_button(
                        label="📄 Download Optimized Resume",
                        data=st.session_state.optimization_result['optimized_resume'],
                        file_name="optimized_resume.pdf",
                        mime="application/pdf",
                        help="Download your resume optimized for this job position",
                        use_container_width=True,
                        key="download_button"
                    )
                
                with analysis_tab:
                    st.markdown("### Full Analysis")
                    st.write(result['analysis'])
                
            except Exception as e:
                st.error(f"An error occurred during resume optimization: {e}")
        else:
            st.warning("Please upload a resume and provide a job description or URL.")
    
with outreach_tab:
    st.header("Outreach Message Generator")
    
    # Initialize session state for outreach
    if 'outreach_result' not in st.session_state:
        st.session_state.outreach_result = None
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'resume_content' not in st.session_state:
        st.session_state.resume_content = None
    if 'outreach_manager' not in st.session_state:
        st.session_state.outreach_manager = OutreachManager()
    
    # Resume upload section
    resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"], key="outreach_resume")
    
    # Store resume content when a new file is uploaded
    if resume_file is not None and (st.session_state.resume_content is None or 
        resume_file.name != st.session_state.get('last_resume_name')):
        st.session_state.resume_content = resume_file.read()
        st.session_state.last_resume_name = resume_file.name
        resume_file.seek(0)  # Reset file pointer after reading
    
    # Job URL input
    job_url = st.text_input("Job Posting URL", 
                          help="Paste the URL of the job posting you're interested in")
    
    # Optional specific interests
    specific_interests = st.text_area(
        "Why are you specifically interested in this role? (Optional)",
        help="Add any specific points about why you're excited about this opportunity"
    )
    
    # Generate initial message
    if st.button("Generate Message") and resume_file and job_url:
        try:
            with st.spinner("Analyzing resume and job posting..."):
                result = st.session_state.outreach_manager.generate_message(
                    resume_file=resume_file,
                    job_url=job_url,
                    specific_interests=specific_interests
                )
                st.session_state.outreach_result = result
                st.session_state.feedback_submitted = False
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Display message and feedback section
    if st.session_state.outreach_result:
        result = st.session_state.outreach_result
        
        if 'error' in result:
            st.error(result['error'])
        else:
            st.success("✅ Message generated successfully!")
            st.subheader("Generated Message")
            
            # Display the message
            message_area = st.text_area(
                "Copy and customize as needed:",
                result['message'],
                height=300,
                key="current_message"
            )
            
            # Feedback section
            st.divider()
            with st.expander("✨ Want to improve this message? Add your feedback!"):
                feedback = st.text_area(
                    "What would you like to change about this message?",
                    placeholder="Example: Make it more casual, focus more on technical skills, etc.",
                    help="Your feedback will be used to generate a new version of the message"
                )
                
                if st.button("Regenerate with Feedback") and feedback:
                    try:
                        with st.spinner("Regenerating message with your feedback..."):
                            # Create feedback dict with previous message and feedback
                            feedback_dict = {
                                'previous_message': result['message'],
                                'feedback': feedback
                            }
                            
                            # Use stored resume content for regeneration
                            from io import BytesIO
                            resume_buffer = BytesIO(st.session_state.resume_content)
                            resume_buffer.name = st.session_state.last_resume_name  # Set name for file type detection
                            
                            # Regenerate message using stored outreach manager
                            new_result = st.session_state.outreach_manager.generate_message(
                                resume_file=resume_buffer,
                                job_url=job_url,
                                specific_interests=specific_interests,
                                user_feedback=feedback_dict
                            )
                            
                            st.session_state.outreach_result = new_result
                            st.session_state.feedback_submitted = True
                            st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred while regenerating: {str(e)}")

with discovery_tab:
    st.header("Job Opening Discovery")
    
    # Initialize session state for job discovery
    if 'job_discovery_manager' not in st.session_state:
        st.session_state.job_discovery_manager = JobDiscoveryManager()
    if 'job_preferences' not in st.session_state:
        st.session_state.job_preferences = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Resume upload for automatic preference extraction
        resume_file = st.file_uploader(
            "Upload Your Resume (Optional)",
            type=["pdf"],
            key="discovery_resume",
            help="Upload your resume to automatically extract job preferences"
        )
        
        # Manual criteria input
        manual_criteria = st.text_area(
            "Additional Job Search Criteria",
            placeholder="Example: Remote work, Series B startups, AI/ML focus, etc.",
            help="Add any specific preferences not in your resume"
        )
        
        # Extract preferences from resume
        if resume_file and st.button("Extract Preferences from Resume"):
            with st.spinner("Analyzing your resume..."):
                preferences = st.session_state.job_discovery_manager.extract_preferences(resume_file)
                if preferences:
                    st.session_state.job_preferences = preferences
                    st.success("✅ Successfully extracted preferences from resume!")
                else:
                    st.error("Could not extract preferences from resume")
    
    with col2:
        if st.session_state.job_preferences:
            st.subheader("Extracted Preferences")
            st.json(st.session_state.job_preferences)
    
    # Search button
    if st.button("Search Job Openings"):
        if not st.session_state.get('job_preferences'):
            st.warning("⚠️ No resume preferences found. Please upload your resume and click 'Extract Preferences from Resume' first for better job matches.")
        try:
            with st.spinner("Searching for relevant job openings..."):
                # Format background from preferences
                background = ""
                if st.session_state.job_preferences:
                    prefs = st.session_state.job_preferences
                    if prefs.get('skills'):
                        background += f"Skills: {', '.join(prefs['skills'])}\n"
                    if prefs.get('industries'):
                        background += f"Industries: {', '.join(prefs['industries'])}\n"
                    if prefs.get('role_level'):
                        background += f"Role Level: {prefs['role_level']}\n"
                    if prefs.get('preferred_companies'):
                        background += f"Preferred Companies: {', '.join(prefs['preferred_companies'])}\n"
                    if prefs.get('education'):
                        edu = prefs['education']
                        background += f"Education: {edu.get('degree', 'N/A')} in {edu.get('field', 'N/A')}"
                else:
                    background = "No resume preferences extracted"
                
                # Search for jobs using both background and criteria
                jobs = st.session_state.job_discovery_manager.search_job_openings(
                    background=background,
                    criteria=manual_criteria or ""
                )
                all_jobs = jobs if jobs else []
                
                # Display results
                st.subheader(f"Found {len(all_jobs)} Relevant Openings")
                
                for job in all_jobs:
                    # Check if we have valid company name and title
                    job_title = job.get('title', 'Unknown Position')
                    company_name = job.get('company', 'Unknown Company')
                    
                    # Create a better formatted expander title
                    expander_title = f"{job_title} at {company_name}"
                    
                    with st.expander(expander_title):
                        # Create two columns for job details
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Display basic job information
                            st.markdown(f"### {job_title}")
                            st.markdown(f"**🏢 Company:** {company_name}")
                            st.markdown(f"**📍 Location:** {job.get('location', 'Unknown Location')}")
                            st.markdown(f"**📅 Posted:** {job.get('posted_date', 'Unknown')}")
                            if job.get('salary'):
                                st.markdown(f"**💰 Salary:** {job.get('salary')}")
                        
                        with col2:
                            # Display application button prominently
                            app_link = job.get('link', 'https://www.linkedin.com/jobs/')
                            st.markdown(f"""
                            <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;'>
                                <a href='{app_link}' target='_blank' style='text-decoration: none;'>
                                    <div style='background-color: #0366d6; color: white; padding: 10px; border-radius: 5px; font-weight: bold;'>
                                        Apply Now
                                    </div>
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display job description
                        st.markdown("### Job Description")
                        st.markdown(job.get('description', 'No description available'))
                        
                        # Display requirements as a bulleted list
                        st.markdown("### Requirements")
                        requirements = job.get('requirements', [])
                        if requirements:
                            for req in requirements:
                                st.markdown(f"- {req}")
                        else:
                            st.markdown("No specific requirements listed")
                        
                        # Get company info
                        company_info = st.session_state.job_discovery_manager.get_company_info(company_name)
                        if company_info:
                            st.markdown("---")
                            st.markdown("### Company Information")
                            
                            # Check if we have multiple companies
                            if "companies" in company_info and isinstance(company_info["companies"], list):
                                # Display each company in a separate section
                                for i, company in enumerate(company_info["companies"]):
                                    st.markdown(f"#### {company.get('name', f'Company {i+1}')}")
                                    for key, value in company.items():
                                        if key != "name":  # Skip name as it's already in the header
                                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                                    if i < len(company_info["companies"]) - 1:
                                        st.markdown("---")
                            else:
                                # Display single company info
                                for key, value in company_info.items():
                                    if key != "name":  # Skip name as it's already in the header
                                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        except Exception as e:
            st.error(f"An error occurred during job search: {str(e)}")

with interview_tab:
    st.header("Interview Preparation")
    
    # Initialize interview prep manager
    if 'interview_prep_manager' not in st.session_state:
        st.session_state.interview_prep_manager = InterviewPrepManager()
    
    # Input for company URL or job posting
    company_url = st.text_input(
        "Enter Company Website or Job Posting URL",
        help="Enter the URL of the company's website or job posting to get reviews and interview details"
    )
    
    # Button to get company info
    if st.button("Get Company Information"):
        try:
            with st.spinner("Fetching company reviews and interview process..."):
                result = st.session_state.interview_prep_manager.get_company_info(company_url)
                
                if result:
                    # Display company reviews
                    st.subheader("🏢 Company Reviews")
                    review = result["company_review"]
                    
                    # Create columns for ratings
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Rating", f"{review['overall_rating']:.1f}/5.0")
                        st.metric("Work-Life Balance", f"{review['work_life_balance']:.1f}/5.0")
                    with col2:
                        st.metric("Compensation", f"{review['compensation']:.1f}/5.0")
                        st.metric("Career Growth", f"{review['career_growth']:.1f}/5.0")
                    with col3:
                        st.metric("Company Culture", f"{review['culture']:.1f}/5.0")
                    
                    # Display pros and cons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("✅ Pros:")
                        for pro in review['pros']:
                            st.write(f"- {pro}")
                    with col2:
                        st.write("⚠️ Cons:")
                        for con in review['cons']:
                            st.write(f"- {con}")
                    
                    # Display additional metrics if any
                    if review['additional_metrics']:
                        st.subheader("Additional Metrics")
                        for metric, value in review['additional_metrics'].items():
                            st.metric(metric.replace("_", " ").title(), f"{value:.1f}/5.0")
                    
                    # Display review sources with citations
                    if review.get('sources'):
                        st.write("")
                        st.markdown("""<div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
                            <p style='font-size: 1rem; color: #333;'><strong>📚 Sources for Company Information:</strong></p>
                            <div style='margin: 0.5rem 0;'></div>
                        """, unsafe_allow_html=True)
                        
                        for i, source in enumerate(review['sources'], 1):
                            st.markdown(f"""<div style='margin-left: 1rem;'>
                                <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
                                    {i}. <a href='{source}' target='_blank' style='color: #1e88e5; text-decoration: underline;'>{source}</a>
                                </p>
                            </div>""", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Display interview process
                    st.subheader("🎯 Interview Process for Data Science/Analytics")
                    interview = result["interview_process"]
                    
                    # Display interview details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Role", interview['role'])
                        st.metric("Difficulty", f"{interview['difficulty']:.1f}/5.0")
                    with col2:
                        st.metric("Duration", interview['duration'])
                    
                    # Display interview stages
                    st.write("📝 Interview Stages:")
                    for i, stage in enumerate(interview['stages'], 1):
                        st.write(f"{i}. {stage}")
                    
                    # Display common questions
                    st.write("❓ Common Technical Questions:")
                    for question in interview['common_questions']:
                        st.write(f"- {question}")
                    
                    # Display tips
                    st.write("💡 Tips from Successful Candidates:")
                    for tip in interview['tips']:
                        st.write(f"- {tip}")
                    
                    # Display interview sources with citations
                    if interview.get('sources'):
                        st.write("")
                        st.markdown("""<div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
                            <p style='font-size: 1rem; color: #333;'><strong>📚 Sources for Interview Information:</strong></p>
                            <div style='margin: 0.5rem 0;'></div>
                        """, unsafe_allow_html=True)
                        
                        for i, source in enumerate(interview['sources'], 1):
                            st.markdown(f"""<div style='margin-left: 1rem;'>
                                <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
                                    {i}. <a href='{source}' target='_blank' style='color: #1e88e5; text-decoration: underline;'>{source}</a>
                                </p>
                            </div>""", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display last updated
                    st.caption(f"Information last updated: {interview['last_updated']}")
                else:
                    st.error("Could not fetch company information. Please try again.")
        except Exception as e:
            st.error(f"Error occurred while fetching company information: {e}")

with debug_tab:
    st.header("Debug & Observability")
    
    # Create subtabs for different debug views
    log_tab, trace_tab, config_tab, search_config_tab = st.tabs([
        "📋 Logs", 
        "🔍 Traces", 
        "⚙️ Configuration",
        "🔎 Search Strategies"
    ])
    
    with log_tab:
        st.subheader("Application Logs")
        
        # Function to load log files
        def load_log_files():
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            if not os.path.exists(log_dir):
                return []
                
            log_files = glob.glob(os.path.join(log_dir, '*.log'))
            return sorted(log_files, key=os.path.getmtime, reverse=True)
        
        log_files = load_log_files()
        
        if not log_files:
            st.info("No log files found. Run some operations to generate logs.")
        else:
            # Select log file
            selected_log = st.selectbox("Select Log File", log_files, format_func=lambda x: os.path.basename(x))
            
            if selected_log:
                # Load log file content
                with open(selected_log, 'r') as f:
                    log_content = f.readlines()
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    log_level = st.selectbox("Log Level", ["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
                with col2:
                    search_term = st.text_input("Search Term")
                with col3:
                    max_lines = st.number_input("Max Lines", min_value=10, max_value=1000, value=100, step=10)
                
                # Filter log content
                filtered_logs = []
                for line in log_content:
                    if log_level != "All" and log_level not in line:
                        continue
                    if search_term and search_term.lower() not in line.lower():
                        continue
                    filtered_logs.append(line)
                
                # Display filtered logs
                filtered_logs = filtered_logs[-max_lines:] if len(filtered_logs) > max_lines else filtered_logs
                
                if filtered_logs:
                    st.text_area("Log Content", "".join(filtered_logs), height=400)
                    
                    # Parse logs into structured format for better analysis
                    try:
                        parsed_logs = []
                        for line in filtered_logs:
                            parts = line.split(' - ', 3)
                            if len(parts) >= 4:
                                timestamp = parts[0]
                                module = parts[1]
                                level = parts[2]
                                message = parts[3].strip()
                                
                                parsed_logs.append({
                                    "timestamp": timestamp,
                                    "module": module,
                                    "level": level,
                                    "message": message
                                })
                        
                        if parsed_logs:
                            st.subheader("Structured Logs")
                            df = pd.DataFrame(parsed_logs)
                            st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not parse logs into structured format: {e}")
                else:
                    st.info("No logs matching the filter criteria.")
    
    with trace_tab:
        st.subheader("LangSmith Traces")
        
        # Check if LangSmith is configured
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "job-platform")
        
        if not langsmith_api_key:
            st.warning("LangSmith API key not configured. Add LANGSMITH_API_KEY to your .env file.")
            
            st.markdown("""
            ### How to set up LangSmith:
            
            1. Sign up for LangSmith at [smith.langchain.com](https://smith.langchain.com/)
            2. Create a new API key in your LangSmith account
            3. Add the following to your `.env` file:
               ```
               LANGSMITH_API_KEY=your_api_key_here
               LANGSMITH_PROJECT=job-platform
               ```
            4. Restart the application
            """)
        else:
            st.success(f"LangSmith configured with project: {langsmith_project}")
            
            # Display LangSmith link
            st.markdown(f"""
            View detailed traces in the [LangSmith Dashboard](https://smith.langchain.com/projects/{langsmith_project}/traces).
            """)
            
            # Show trace summary if available
            if 'job_discovery_manager' in st.session_state and hasattr(st.session_state.job_discovery_manager, 'langsmith_client'):
                st.info("This section will display a summary of recent traces from your job search operations.")
                
                if st.button("Load Recent Traces"):
                    with st.spinner("Loading trace data..."):
                        st.info("In a production app, we would display recent trace summaries here using the LangSmith API.")
            else:
                st.info("Initialize the Job Discovery component to enable trace collection.")
    
    with config_tab:
        st.subheader("Environment Configuration")
        
        # Show environment variables (excluding sensitive ones)
        env_vars = {
            "OPENAI_API_KEY": "********" if os.getenv("OPENAI_API_KEY") else "Not Set",
            "PERPLEXITY_API_KEY": "********" if os.getenv("PERPLEXITY_API_KEY") else "Not Set",
            "LANGSMITH_API_KEY": "********" if os.getenv("LANGSMITH_API_KEY") else "Not Set",
            "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "Not Set"),
            "PYTHONPATH": os.getenv("PYTHONPATH", "Not Set"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
        }
        
        st.json(env_vars)
        
        # Display logging configuration options
        st.subheader("Logging Configuration")
        
        log_level = st.selectbox(
            "Application Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=1  # Default to INFO
        )
        
        enable_request_logging = st.checkbox("Log Full API Requests", value=False, 
            help="Enable to log the full request and response payloads (may contain sensitive data)")
        
        if st.button("Update Logging Configuration"):
            # In a production app, this would update the logging configuration
            st.success("Logging configuration updated. Changes will take effect after restart.")
            
            # For demo purposes, show what would be updated
            new_config = {
                "LOG_LEVEL": log_level,
                "LOG_FULL_REQUESTS": enable_request_logging
            }
            
            st.json(new_config)

    with search_config_tab:
        st.subheader("Job Search Strategy Configuration")
        
        if 'job_discovery_manager' not in st.session_state:
            st.session_state.job_discovery_manager = JobDiscoveryManager()
        
        # Get search configurations
        search_configs = st.session_state.job_discovery_manager.search_configs
        active_config = st.session_state.job_discovery_manager.active_config_name
        
        # Display configuration selector and metrics
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Select configuration
            selected_config = st.selectbox(
                "Select Search Configuration",
                options=list(search_configs.keys()),
                index=list(search_configs.keys()).index(active_config) if active_config in search_configs else 0,
                key="search_config_selector"
            )
            
            # Set as active button
            if st.button("Set as Active Configuration"):
                st.session_state.job_discovery_manager.active_config_name = selected_config
                st.success(f"Configuration '{selected_config}' set as active")
                
            # Create new configuration button
            if st.button("Create New Configuration"):
                st.session_state["creating_new_config"] = True
        
        with col2:
            # Display configuration metrics
            if selected_config in search_configs:
                config = search_configs[selected_config]
                
                st.subheader(f"Configuration: {selected_config}")
                st.markdown(f"**Description:** {config.get('description', 'No description')}")
                
                # Create metrics columns
                metric_cols = st.columns(4)
                metrics = config.get("metrics", {})
                
                with metric_cols[0]:
                    st.metric("Total Runs", metrics.get("total_runs", 0))
                with metric_cols[1]:
                    success_rate = f"{(metrics.get('successful_runs', 0) / metrics.get('total_runs', 1) * 100):.1f}%" if metrics.get("total_runs", 0) > 0 else "N/A"
                    st.metric("Success Rate", success_rate)
                with metric_cols[2]:
                    st.metric("Avg. Jobs Found", f"{metrics.get('average_jobs_returned', 0):.1f}")
                with metric_cols[3]:
                    st.metric("Avg. Response Time", f"{metrics.get('average_response_time', 0):.2f}s")
                
                # Last updated
                st.caption(f"Last updated: {config.get('updated_at', 'Never')}")
        
        # Configuration editor
        st.markdown("---")
        
        if "creating_new_config" in st.session_state and st.session_state["creating_new_config"]:
            st.subheader("Create New Search Configuration")
            
            new_config_name = st.text_input("Configuration Name", key="new_config_name")
            new_config_description = st.text_area("Description", key="new_config_description")
            
            # Model and temperature
            col1, col2 = st.columns(2)
            with col1:
                new_config_model = st.selectbox("Model", ["sonar", "poe", "claude-3-opus-20240229"], key="new_config_model")
            with col2:
                new_config_temp = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1, key="new_config_temp")
            
            # Prompts
            new_config_system_prompt = st.text_area(
                "System Prompt", 
                height=200,
                key="new_config_system_prompt"
            )
            
            new_config_user_prompt = st.text_area(
                "User Prompt Template", 
                height=200,
                key="new_config_user_prompt",
                help="Use {background} and {criteria} as placeholders for the candidate background and search criteria"
            )
            
            # Options
            new_config_use_fallback = st.checkbox("Use Fallback Search", value=True, key="new_config_use_fallback")
            
            # Save buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Configuration"):
                    if not new_config_name:
                        st.error("Configuration name is required")
                    elif new_config_name in search_configs:
                        st.error(f"Configuration '{new_config_name}' already exists")
                    elif not new_config_system_prompt or not new_config_user_prompt:
                        st.error("System prompt and user prompt are required")
                    else:
                        # Create new configuration
                        new_config = {
                            "description": new_config_description or f"Custom configuration created on {datetime.now().strftime('%Y-%m-%d')}",
                            "system_prompt": new_config_system_prompt,
                            "user_prompt_template": new_config_user_prompt,
                            "model": new_config_model,
                            "temperature": new_config_temp,
                            "use_fallback": new_config_use_fallback,
                            "metrics": {
                                "total_runs": 0,
                                "successful_runs": 0,
                                "average_jobs_returned": 0,
                                "average_response_time": 0
                            },
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat()
                        }
                        
                        # Add to configurations
                        search_configs[new_config_name] = new_config
                        
                        # Save to file
                        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'search_configs.json')
                        try:
                            # Create config directory if it doesn't exist
                            config_dir = os.path.dirname(config_path)
                            if not os.path.exists(config_dir):
                                os.makedirs(config_dir)
                                
                            with open(config_path, 'w') as f:
                                json.dump(search_configs, f, indent=2)
                            st.success(f"Configuration '{new_config_name}' created successfully")
                            st.session_state["creating_new_config"] = False
                        except Exception as e:
                            st.error(f"Error saving configuration: {e}")
            
            with col2:
                if st.button("Cancel"):
                    st.session_state["creating_new_config"] = False
        else:
            # Configuration details
            if selected_config in search_configs:
                config = search_configs[selected_config]
                
                st.subheader("Configuration Details")
                
                # Editable fields
                edited_config = {}
                edited_config["description"] = st.text_area(
                    "Description", 
                    value=config.get("description", ""),
                    key=f"edit_description_{selected_config}"
                )
                
                # Model and temperature
                col1, col2 = st.columns(2)
                with col1:
                    edited_config["model"] = st.selectbox(
                        "Model", 
                        ["sonar", "poe", "claude-3-opus-20240229"],
                        index=["sonar", "poe", "claude-3-opus-20240229"].index(config.get("model", "sonar")),
                        key=f"edit_model_{selected_config}"
                    )
                with col2:
                    edited_config["temperature"] = st.slider(
                        "Temperature", 
                        0.0, 1.0, float(config.get("temperature", 0.5)), 0.1,
                        key=f"edit_temp_{selected_config}"
                    )
                
                # Prompts
                st.markdown("### System Prompt")
                edited_config["system_prompt"] = st.text_area(
                    "", 
                    value=config.get("system_prompt", ""),
                    height=200,
                    key=f"edit_system_prompt_{selected_config}",
                    label_visibility="collapsed"
                )
                
                st.markdown("### User Prompt Template")
                st.caption("Use {background} and {criteria} as placeholders for the candidate background and search criteria")
                edited_config["user_prompt_template"] = st.text_area(
                    "", 
                    value=config.get("user_prompt_template", ""),
                    height=200,
                    key=f"edit_user_prompt_{selected_config}",
                    label_visibility="collapsed"
                )
                
                # Options
                edited_config["use_fallback"] = st.checkbox(
                    "Use Fallback Search", 
                    value=config.get("use_fallback", True),
                    key=f"edit_use_fallback_{selected_config}"
                )
                
                # Update button
                if st.button("Update Configuration"):
                    # Update configuration
                    for key, value in edited_config.items():
                        search_configs[selected_config][key] = value
                    
                    search_configs[selected_config]["updated_at"] = datetime.now().isoformat()
                    
                    # Save to file
                    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'search_configs.json')
                    try:
                        # Create config directory if it doesn't exist
                        config_dir = os.path.dirname(config_path)
                        if not os.path.exists(config_dir):
                            os.makedirs(config_dir)
                            
                        with open(config_path, 'w') as f:
                            json.dump(search_configs, f, indent=2)
                        st.success(f"Configuration '{selected_config}' updated successfully")
                    except Exception as e:
                        st.error(f"Error saving configuration: {e}")
                
                # Test configuration section
                st.markdown("---")
                st.subheader("Test Configuration")
                
                test_background = st.text_area(
                    "Test Background",
                    placeholder="Enter candidate background information",
                    key=f"test_background_{selected_config}"
                )
                
                test_criteria = st.text_area(
                    "Test Criteria",
                    placeholder="Enter additional search criteria",
                    key=f"test_criteria_{selected_config}"
                )
                
                if st.button("Run Test Search"):
                    if not test_background:
                        st.error("Background information is required for testing")
                    else:
                        with st.spinner(f"Running test search with configuration '{selected_config}'..."):
                            test_jobs = st.session_state.job_discovery_manager.search_job_openings(
                                background=test_background,
                                criteria=test_criteria,
                                config_name=selected_config
                            )
                            
                            # Store results in session state for display
                            st.session_state[f"test_results_{selected_config}"] = test_jobs
                
                # Display test results if available
                if f"test_results_{selected_config}" in st.session_state:
                    test_jobs = st.session_state[f"test_results_{selected_config}"]
                    st.markdown(f"### Test Results: {len(test_jobs)} jobs found")
                    
                    for i, job in enumerate(test_jobs):
                        with st.expander(f"{i+1}. {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}"):
                            st.json(job)
