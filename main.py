import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from modules.resume_optimizer import ResumeOptimizer
from modules.outreach import OutreachManager
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
st.title("Job Hunt Assistant")

# Create a sidebar for navigation
page = st.sidebar.selectbox(
    "Select a Tool",
    ["Resume Optimization", "Outreach"]
)

if page == "Resume Optimization":
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
    
elif page == "Outreach":
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
