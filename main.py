import streamlit as st
from modules.outreach import OutreachManager
from modules.resume_optimizer import ResumeOptimizer

st.set_page_config(page_title="Job Hunt Assistant", layout="wide")

def main():
    st.title("Job Hunt Assistant")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select a feature:",
            ["Resume Optimization", "Outreach"]
        )
    
    if page == "Resume Optimization":
        st.header("Resume Optimization")
        resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
        job_description = st.text_area("Paste the job description here")
        
        if st.button("Analyze Resume") and resume_file and job_description:
            optimizer = ResumeOptimizer()
            with st.spinner("Analyzing your resume..."):
                analysis = optimizer.analyze_resume(resume_file, job_description)
                st.subheader("Analysis Results")
                st.write(analysis)
    
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

if __name__ == "__main__":
    main()
