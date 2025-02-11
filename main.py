import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from modules.resume_optimizer import ResumeOptimizer
from modules.outreach import OutreachManager
from modules.job_discovery import JobDiscoveryManager
from modules.interview_prep import InterviewPrepManager
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv(find_dotenv(), override=True)

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
resume_tab, outreach_tab, discovery_tab, interview_tab = st.tabs(["📝 Resume Optimization", "✉️ Outreach", "🔍 Job Discovery", "🎯 Interview Prep"])

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
                    with st.expander(f"{job['title']} at {job['company']}"):
                        st.write(f"🏢 **Company:** {job['company']}")
                        st.write(f"📍 **Location:** {job['location']}")
                        st.write(f"📅 **Posted:** {job['posted_date']}")
                        
                        # Get company info
                        company_info = st.session_state.job_discovery_manager.get_company_info(job['company'])
                        if company_info:
                            st.write("---")
                            st.write("### Company Information")
                            for key, value in company_info.items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        st.write("---")
                        st.write("### Job Description")
                        st.write(job['description'])
                        
                        st.write("### Requirements")
                        st.write(job['requirements'])
                        
                        st.markdown(f"[🔗 Apply Now]({job['link']})")
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
