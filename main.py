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
                # Perform resume optimization
                optimizer = ResumeOptimizer()
                result = optimizer.analyze_resume(
                    resume_file, 
                    job_description=job_description, 
                    job_url=job_url
                )
                
                # Display Optimization Summary
                st.subheader("Optimization Insights")
                st.markdown("**Key Recommended Changes:**")
                st.markdown(result['changes_summary'])
                
                # Detailed Analysis (optional, can be expanded/collapsed)
                with st.expander("Detailed Optimization Analysis"):
                    st.write(result['analysis'])
                
                # Download Optimized Resume
                st.download_button(
                    label="Download Optimized Resume",
                    data=result['optimized_resume'],
                    file_name="optimized_resume.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"An error occurred during resume optimization: {e}")
        else:
            st.warning("Please upload a resume and provide a job description or URL.")
    
elif page == "Outreach":
    st.header("Outreach Message Generator")
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name")
        contact_name = st.text_input("Contact Name (if known)")
        role = st.text_input("Role you're interested in")
    
    with col2:
        your_background = st.text_area("Brief description of your background")
        specific_interests = st.text_area("Why are you interested in this company/role?")
    
    if st.button("Generate Message") and company_name and role:
        outreach = OutreachManager()
        with st.spinner("Generating personalized message..."):
            message = outreach.generate_message(
                company_name=company_name,
                contact_name=contact_name,
                role=role,
                background=your_background,
                interests=specific_interests
            )
            st.subheader("Generated Message")
            st.text_area("Copy and customize as needed:", message, height=300)
