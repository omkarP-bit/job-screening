from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import traceback
from datetime import datetime

# Import the complete JobScreeningSystem and related classes
import re
import numpy as np
import pandas as pd
import spacy
import PyPDF2
import docx
import pdfplumber
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spacy English model: python -m spacy download en_core_web_sm")
    nlp = None

class JobDescriptionParser:
    """Extracts key requirements from job descriptions"""
    
    def __init__(self):
        pass
    
    def extract_text(self, file_path):
        """Extract text from JD files in various formats"""
        if file_path.endswith('.pdf'):
            return self._extract_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._extract_from_docx(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError("Unsupported file format")
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"Fallback PDF extraction also failed: {e2}")
        return text
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting from DOCX: {e}")
            return ""
    
    def summarize(self, text):
        """Generate simple summary of job description"""
        sentences = text.split('.')
        summary = '. '.join(sentences[:3]) + '.'
        return summary
    
    def extract_requirements(self, text):
        """Extract key requirements from job description"""
        if not nlp:
            return self._extract_requirements_basic(text)
        
        doc = nlp(text)
        
        skills = []
        education = []
        experience = []
        
        skill_patterns = ["skills", "proficiency", "knowledge of", "familiarity with", "expertise in"]
        edu_patterns = ["degree", "qualification", "education", "graduate", "bachelor", "master", "phd"]
        exp_patterns = ["experience", "years", "worked", "background in"]
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            if any(pattern in sent_text for pattern in skill_patterns):
                skills.append(sent.text.strip())
            
            if any(pattern in sent_text for pattern in edu_patterns):
                education.append(sent.text.strip())
            
            if any(pattern in sent_text for pattern in exp_patterns):
                experience.append(sent.text.strip())
        
        location_pattern = r"location:?\s*([\w\s,]+)"
        job_type_pattern = r"(full[- ]time|part[- ]time|contract|remote|hybrid)"
        
        location_match = re.search(location_pattern, text, re.IGNORECASE)
        job_type_match = re.search(job_type_pattern, text, re.IGNORECASE)
        
        location = location_match.group(1).strip() if location_match else "Not specified"
        job_type = job_type_match.group(1).strip() if job_type_match else "Not specified"
        
        requirements = {
            "skills": skills,
            "education": education,
            "experience": experience,
            "location": location,
            "job_type": job_type,
            "summary": self.summarize(text)
        }
        
        return requirements
    
    def _extract_requirements_basic(self, text):
        """Basic requirements extraction without spaCy"""
        text_lower = text.lower()
        
        skills = []
        skill_keywords = ["python", "java", "javascript", "sql", "html", "css", "react", "angular", "node"]
        for skill in skill_keywords:
            if skill in text_lower:
                skills.append(skill)
        
        education = []
        if "bachelor" in text_lower or "degree" in text_lower:
            education.append("Bachelor's degree required")
        if "master" in text_lower:
            education.append("Master's degree preferred")
        
        experience = []
        exp_match = re.search(r"(\d+)\s*years?\s*experience", text_lower)
        if exp_match:
            experience.append(f"{exp_match.group(1)} years of experience required")
        
        return {
            "skills": skills,
            "education": education,
            "experience": experience,
            "location": "Not specified",
            "job_type": "Not specified",
            "summary": self.summarize(text)
        }


class ResumeParser:
    """Extracts structured information from candidate resumes"""
    
    def __init__(self):
        self.nlp = nlp
    
    def extract_text(self, file_path):
        """Extract text from resume in various formats"""
        if file_path.endswith('.pdf'):
            return self._extract_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._extract_from_docx(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError("Unsupported file format")
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"Fallback PDF extraction also failed: {e2}")
        return text
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting from DOCX: {e}")
            return ""
    
    def extract_contact_info(self, text):
        """Extract contact information from resume"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        phone_pattern = r'(\+\d{1,3}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}'
        phones = re.findall(phone_pattern, text)
        
        name = ""
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text
                    break
        
        if not name:
            lines = text.strip().split('\n')
            if lines:
                first_line = lines[0].strip()
                if len(first_line.split()) <= 3 and not any(char.isdigit() for char in first_line):
                    name = first_line
        
        return {
            "name": name,
            "email": emails[0] if emails else "",
            "phone": ''.join(phones[0]) if phones else ""
        }
    
    def extract_education(self, text):
        """Extract education information from resume"""
        education = []
        edu_keywords = ["degree", "bachelor", "master", "phd", "btech", "mtech", "b.tech", "m.tech", "diploma"]
        
        if self.nlp:
            doc = self.nlp(text)
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if any(keyword in sent_text for keyword in edu_keywords):
                    education.append(sent.text.strip())
        else:
            lines = text.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in edu_keywords):
                    education.append(line.strip())
        
        return education
    
    def extract_skills(self, text):
        """Extract skills from resume"""
        tech_skills = [
            "python", "java", "javascript", "sql", "html", "css", "react", "angular", "node", 
            "aws", "azure", "gcp", "docker", "kubernetes", "git", "tensorflow", "pytorch", 
            "machine learning", "data science", "deep learning", "ai", "artificial intelligence",
            "nlp", "natural language processing", "computer vision", "c++", "c#", "php", "r"
        ]
        
        skills = []
        text_lower = text.lower()
        
        for skill in tech_skills:
            if skill in text_lower:
                skills.append(skill)
        
        return skills
    
    def extract_experience(self, text):
        """Extract work experience from resume"""
        experience = []
        exp_keywords = ["experience", "work", "employment", "job", "position", "role"]
        
        if self.nlp:
            doc = self.nlp(text)
            
            exp_section = None
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in exp_keywords) and len(line) < 30:
                    exp_section = i
                    break
            
            if exp_section is not None:
                exp_text = '\n'.join(lines[exp_section+1:])
                for sent in self.nlp(exp_text).sents:
                    for ent in sent.ents:
                        if ent.label_ == "ORG":
                            experience.append(sent.text.strip())
                            break
        else:
            lines = text.split('\n')
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in exp_keywords):
                    experience.append(line.strip())
        
        return experience
    
    def parse_resume(self, file_path):
        """Parse a resume file and extract structured information"""
        text = self.extract_text(file_path)
        
        contact_info = self.extract_contact_info(text)
        education = self.extract_education(text)
        skills = self.extract_skills(text)
        experience = self.extract_experience(text)
        
        return {
            "contact_info": contact_info,
            "education": education,
            "skills": skills,
            "experience": experience,
            "full_text": text
        }


class MatchingEngine:
    """Implements algorithms for matching resumes against job requirements"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    def calculate_skill_match(self, jd_skills, resume_skills):
        """Calculate match score for skills"""
        if not jd_skills:
            return 1.0
        
        jd_skill_keywords = []
        for skill_text in jd_skills:
            words = skill_text.lower().split()
            jd_skill_keywords.extend(words)
        
        matches = 0
        for skill in jd_skill_keywords:
            if any(skill in resume_skill.lower() for resume_skill in resume_skills):
                matches += 1
        
        return matches / len(jd_skill_keywords) if jd_skill_keywords else 0.0
    
    def calculate_experience_match(self, jd_experience, resume_experience):
        """Calculate match score for experience using semantic similarity"""
        if not jd_experience or not resume_experience:
            return 0.0
            
        jd_exp_text = ' '.join(jd_experience)
        resume_exp_text = ' '.join(resume_experience)
        
        try:
            corpus = [jd_exp_text, resume_exp_text]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            print(f"Error calculating experience match: {e}")
            return 0.0
    
    def calculate_education_match(self, jd_education, resume_education):
        """Calculate match score for education"""
        if not jd_education:
            return 1.0
            
        if not resume_education:
            return 0.0
        
        try:
            jd_edu_text = ' '.join(jd_education)
            resume_edu_text = ' '.join(resume_education)
            
            corpus = [jd_edu_text, resume_edu_text]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            print(f"Error calculating education match: {e}")
            return 0.0
    
    def match_resume_to_jd(self, job_requirements, resume_data, weights=None):
        """Calculate overall match score between resume and job description"""
        if weights is None:
            weights = {
                "skills": 0.5,
                "experience": 0.3,
                "education": 0.2
            }
        
        skill_score = self.calculate_skill_match(
            job_requirements.get("skills", []),
            resume_data.get("skills", [])
        )
        
        experience_score = self.calculate_experience_match(
            job_requirements.get("experience", []),
            resume_data.get("experience", [])
        )
        
        education_score = self.calculate_education_match(
            job_requirements.get("education", []),
            resume_data.get("education", [])
        )
        
        overall_score = (
            skill_score * weights["skills"] +
            experience_score * weights["experience"] +
            education_score * weights["education"]
        )
        
        match_details = {
            "overall_score": overall_score,
            "skill_score": skill_score,
            "experience_score": experience_score,
            "education_score": education_score,
            "matched_skills": []
        }
        
        jd_skills = job_requirements.get("skills", [])
        resume_skills = resume_data.get("skills", [])
        
        for jd_skill in jd_skills:
            jd_skill_words = jd_skill.lower().split()
            for word in jd_skill_words:
                if any(word in resume_skill.lower() for resume_skill in resume_skills):
                    match_details["matched_skills"].append(word)
        
        return match_details


class JobScreeningSystem:
    """Main system class that coordinates the job screening process"""
    
    def __init__(self, smtp_config=None):
        self.jd_parser = JobDescriptionParser()
        self.resume_parser = ResumeParser()
        self.matching_engine = MatchingEngine()
        
        self.job_requirements = None
        self.candidates = []
        self.match_results = []
        
    def load_job_description(self, file_path):
        """Load and parse job description"""
        text = self.jd_parser.extract_text(file_path)
        self.job_requirements = self.jd_parser.extract_requirements(text)
        
        filename = os.path.basename(file_path)
        job_title = os.path.splitext(filename)[0]
        self.job_requirements["title"] = job_title
        
        return self.job_requirements
    
    def load_resumes(self, directory_path):
        """Load and parse all resumes in directory"""
        self.candidates = []
        
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist")
            return []
        
        for filename in os.listdir(directory_path):
            if filename.endswith(('.pdf', '.docx', '.txt')):
                file_path = os.path.join(directory_path, filename)
                try:
                    resume_data = self.resume_parser.parse_resume(file_path)
                    resume_data["file_name"] = filename
                    self.candidates.append(resume_data)
                    print(f"Successfully parsed: {filename}")
                except Exception as e:
                    print(f"Error parsing resume {filename}: {e}")
        
        return self.candidates
    
    def screen_candidates(self, threshold=0.8, weights=None):
        """Screen all candidates against job requirements"""
        if not self.job_requirements:
            raise ValueError("Job requirements not loaded")
        
        if not self.candidates:
            raise ValueError("No candidates loaded")
        
        self.match_results = []
        
        for candidate in self.candidates:
            match_details = self.matching_engine.match_resume_to_jd(
                self.job_requirements,
                candidate,
                weights
            )
            
            result = {
                "candidate": candidate["contact_info"]["name"] or "Unknown",
                "email": candidate["contact_info"]["email"],
                "file_name": candidate["file_name"],
                "match_score": match_details["overall_score"],
                "shortlisted": match_details["overall_score"] >= threshold,
                "details": match_details
            }
            
            self.match_results.append(result)
        
        self.match_results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return self.match_results


# Flask Application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'job_descriptions'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'resumes'), exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global screening system instance
screening_system = JobScreeningSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_job_description', methods=['POST'])
def upload_job_description():
    try:
        if 'job_description' not in request.files:
            return jsonify({'error': 'No job description file provided'}), 400
        
        file = request.files['job_description']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_descriptions', filename)
            file.save(file_path)
            
            # Parse job description using the complete system
            job_requirements = screening_system.load_job_description(file_path)
            
            return jsonify({
                'message': 'Job description uploaded and parsed successfully',
                'job_requirements': job_requirements,
                'filename': filename
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Error processing job description: {str(e)}'}), 500

@app.route('/upload_resumes', methods=['POST'])
def upload_resumes():
    try:
        if 'resumes' not in request.files:
            return jsonify({'error': 'No resume files provided'}), 400
        
        files = request.files.getlist('resumes')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        resume_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'resumes')
        
        # Clear previous resumes
        for old_file in os.listdir(resume_dir):
            os.remove(os.path.join(resume_dir, old_file))
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                
                file_path = os.path.join(resume_dir, filename)
                file.save(file_path)
                uploaded_files.append(filename)
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        # Parse resumes using the complete system
        candidates = screening_system.load_resumes(resume_dir)
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} resume(s)',
            'uploaded_files': uploaded_files,
            'candidates_parsed': len(candidates)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing resumes: {str(e)}'}), 500

@app.route('/screen_candidates', methods=['POST'])
def screen_candidates():
    try:
        data = request.get_json() or {}
        threshold = float(data.get('threshold', 0.8))
        
        weights = data.get('weights', {
            'skills': 0.5,
            'experience': 0.3,
            'education': 0.2
        })
        
        if not screening_system.job_requirements:
            return jsonify({'error': 'No job description loaded. Please upload a job description first.'}), 400
        
        if not screening_system.candidates:
            return jsonify({'error': 'No candidates loaded. Please upload resumes first.'}), 400
        
        # Perform screening
        results = screening_system.screen_candidates(threshold, weights)
        
        response = {
            'message': 'Screening completed successfully',
            'total_candidates': len(results),
            'shortlisted_candidates': len([r for r in results if r.get('shortlisted', False)]),
            'threshold': threshold,
            'results': results
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Error during screening: {str(e)}'}), 500

@app.route('/get_results')
def get_results():
    try:
        if not screening_system.match_results:
            return jsonify({'error': 'No screening results available'}), 400
        
        return jsonify({
            'results': screening_system.match_results,
            'total_candidates': len(screening_system.match_results),
            'shortlisted_candidates': len([r for r in screening_system.match_results if r.get('shortlisted', False)])
        })
    
    except Exception as e:
        return jsonify({'error': f'Error retrieving results: {str(e)}'}), 500

@app.route('/send_notifications', methods=['POST'])
def send_notifications():
    try:
        data = request.get_json() or {}
        company_name = data.get('company_name', 'Our Company')
        
        if not screening_system.match_results:
            return jsonify({'error': 'No screening results available'}), 400
        
        notification_results = []
        for result in screening_system.match_results:
            notification_results.append({
                'candidate': result.get('candidate', 'Unknown'),
                'email': result.get('email', 'N/A'),
                'notification_sent': True,  # Simulated
                'shortlisted': result.get('shortlisted', False)
            })
        
        return jsonify({
            'message': 'Notifications sent successfully',
            'notification_results': notification_results
        })
    
    except Exception as e:
        return jsonify({'error': f'Error sending notifications: {str(e)}'}), 500

@app.route('/download_results')
def download_results():
    try:
        if not screening_system.match_results:
            return jsonify({'error': 'No results available to download'}), 400
        
        results_dir = app.config['UPLOAD_FOLDER']
        results_filename = 'screening_results.json'
        results_file_path = os.path.join(results_dir, results_filename)
        
        with open(results_file_path, 'w', encoding='utf-8') as f:
            json.dump(screening_system.match_results, f, indent=2, ensure_ascii=False)
        
        return send_from_directory(directory=results_dir, path=results_filename, as_attachment=True, download_name='screening_results.json')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error saving results: {str(e)}'}), 500

@app.route('/reset_system', methods=['POST'])
def reset_system():
    try:
        global screening_system
        screening_system = JobScreeningSystem()
        
        for folder in ['job_descriptions', 'resumes']:
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
        
        return jsonify({'message': 'System reset successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Error resetting system: {str(e)}'}), 500

@app.route('/status')
def get_status():
    try:
        status = {
            'job_description_loaded': screening_system.job_requirements is not None,
            'candidates_loaded': len(screening_system.candidates),
            'screening_completed': len(screening_system.match_results) > 0,
            'job_requirements': screening_system.job_requirements,
            'last_screening_time': datetime.now().isoformat()
        }
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({'error': f'Error getting status: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)