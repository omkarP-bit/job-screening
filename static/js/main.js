document.addEventListener('DOMContentLoaded', () => {
    const jobDescriptionInput = document.getElementById('jobDescription');
    const jobUploadArea = document.getElementById('jobUploadArea');
    const jobFilePreview = document.getElementById('jobFilePreview');
    const jobStatus = document.getElementById('jobStatus');

    const resumeInput = document.getElementById('resumes');
    const resumeUploadArea = document.getElementById('resumeUploadArea');
    const resumeFilePreview = document.getElementById('resumeFilePreview');
    const resumeStatus = document.getElementById('resumeStatus');

    const thresholdSlider = document.getElementById('threshold');
    const thresholdValueDisplay = document.getElementById('thresholdValue');
    const skillsWeightSlider = document.getElementById('skillsWeight');
    const experienceWeightSlider = document.getElementById('experienceWeight');
    const educationWeightSlider = document.getElementById('educationWeight');
    // const keywordsWeightSlider = document.getElementById('keywordsWeight'); // Not used by current backend

    const screenBtn = document.getElementById('screenBtn');
    const exportPdfBtn = document.getElementById('exportPdfBtn'); // Specific PDF export
    const exportCsvBtn = document.getElementById('exportCsvBtn'); // Specific CSV export
    const resetBtn = document.getElementById('resetBtn');

    const resultsSection = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingTitle = document.getElementById('loadingTitle');
    const loadingMessage = document.getElementById('loadingMessage');
    const autoNotificationStatusDiv = document.getElementById('autoNotificationStatus');
    const autoNotificationStatusText = document.getElementById('autoNotificationStatusText');

    let jobFileUploaded = false;
    let resumeFilesUploaded = false;
    let screeningResultsAvailable = false; // New flag

    // --- Popup Notification Function (Centered) ---
    function showPopupNotification(title, message, type = 'info') {
        const popupContainer = document.getElementById('popupNotificationContainer');
        if (!popupContainer) {
            console.warn('Popup notification container not found!');
            return;
        }
        // Hide results section if visible
        if (resultsSection && resultsSection.style.display !== 'none') {
            resultsSection.style.display = 'none';
        }
        // Hide loading overlay if visible
        if (loadingOverlay && loadingOverlay.style.display !== 'none') {
            loadingOverlay.style.display = 'none';
        }
        const iconClass = {
            success: 'fa-check-circle',
            error: 'fa-times-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        }[type] || 'fa-info-circle';
        const popup = document.createElement('div');
        popup.className = `popup-notification ${type}`;
        popup.innerHTML = `
            <span class="popup-icon"><i class="fas ${iconClass}"></i></span>
            <div class="popup-content">
                <div class="popup-title" style="font-weight:700;font-size:1.2em;">${title}</div>
                <div class="popup-message">${message}</div>
            </div>
            <button class="popup-close" aria-label="Close">&times;</button>
        `;
        popup.querySelector('.popup-close').onclick = () => popup.remove();
        popupContainer.appendChild(popup);
        setTimeout(() => { popup.remove(); }, 5000);
    }

    // --- Generic File Upload Function ---
    async function uploadFile(file, url, previewElement, statusElement, inputName, isMultiple = false) {
        const formData = new FormData();
        if (isMultiple) {
            // For multiple files, 'file' is a FileList
            for (let i = 0; i < file.length; i++) {
                formData.append(inputName, file[i]);
            }
        } else {
            formData.append(inputName, file);
        }

        statusElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        previewElement.classList.add('show');
        if (isMultiple) {
            previewElement.innerHTML = `<p>Uploading ${file.length} file(s)...</p>`;
        } else {
            previewElement.innerHTML = `<p>Uploading ${file.name}...</p>`;
        }

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
            });
            const result = await response.json();

            if (response.ok) {
                let successMsg = '';
                if (isMultiple) {
                    successMsg = `<p><i class="fas fa-check-circle" style="color: var(--success-color);"></i> ${result.uploaded_files ? result.uploaded_files.length : file.length} resume(s) uploaded successfully.</p>`;
                    if(result.candidates_parsed !== undefined) {
                        successMsg += `<p><i class="fas fa-info-circle"></i> Parsed ${result.candidates_parsed} candidates.</p>`;
                    }
                    resumeFilesUploaded = true;
                } else {
                    successMsg = `<p><i class="fas fa-check-circle" style="color: var(--success-color);"></i> ${file.name} uploaded successfully.</p>`;
                    // console.log('Job Requirements:', result.job_requirements);
                    jobFileUploaded = true;
                }
                previewElement.innerHTML = successMsg;
                statusElement.innerHTML = '<i class="fas fa-check-circle" style="color: var(--success-color);"></i>';
                showPopupNotification('Success', result.message || 'File(s) uploaded successfully!', 'success');
            } else {
                previewElement.innerHTML = `<p><i class="fas fa-times-circle" style="color: var(--error-color);"></i> Upload failed: ${result.error || 'Unknown server error'}</p>`;
                statusElement.innerHTML = '<i class="fas fa-times-circle" style="color: var(--error-color);"></i>';
                showPopupNotification('Error', result.error || 'File(s) upload failed.', 'error');
                if (inputName === 'job_description') jobFileUploaded = false;
                if (inputName === 'resumes') resumeFilesUploaded = false;
            }
        } catch (error) {
            console.error('Upload error:', error);
            previewElement.innerHTML = `<p><i class="fas fa-exclamation-triangle" style="color: var(--error-color);"></i> Network error or issue during upload. Check console.</p>`;
            statusElement.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: var(--error-color);"></i>';
            showPopupNotification('Error', 'An unexpected error occurred during upload.', 'error');
            if (inputName === 'job_description') jobFileUploaded = false;
            if (inputName === 'resumes') resumeFilesUploaded = false;
        }
        updateScreenButtonState();
    }

    function updateScreenButtonState() {
        if (jobFileUploaded && resumeFilesUploaded) {
            screenBtn.disabled = false; // Enable screen button if files are ready
        } else {
            screenBtn.disabled = true;
        }
        // Export and Notify buttons depend on results being available
        if (exportPdfBtn) exportPdfBtn.disabled = !screeningResultsAvailable;
        if (exportCsvBtn) exportCsvBtn.disabled = !screeningResultsAvailable;
    }

    // --- Event Listeners for Job Description ---
    if (jobDescriptionInput && jobUploadArea && jobFilePreview && jobStatus) {
        jobDescriptionInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                uploadFile(file, '/upload_job_description', jobFilePreview, jobStatus, 'job_description');
            }
        });

        setupDragAndDrop(jobUploadArea, (files) => {
            if (files && files.length > 0) {
                jobDescriptionInput.files = files; // Assign to input to trigger change or for consistency
                uploadFile(files[0], '/upload_job_description', jobFilePreview, jobStatus, 'job_description');
            }
        });
    } else {
        console.error('One or more job description DOM elements are missing.');
    }

    // --- Event Listeners for Resumes ---
    if (resumeInput && resumeUploadArea && resumeFilePreview && resumeStatus) {
        resumeInput.addEventListener('change', (event) => {
            const files = event.target.files;
            if (files.length > 0) {
                uploadFile(files, '/upload_resumes', resumeFilePreview, resumeStatus, 'resumes', true);
            }
        });

        setupDragAndDrop(resumeUploadArea, (files) => {
            if (files && files.length > 0) {
                resumeInput.files = files; // Assign to input
                uploadFile(files, '/upload_resumes', resumeFilePreview, resumeStatus, 'resumes', true);
            }
        });
    } else {
        console.error('One or more resume DOM elements are missing.');
    }

    // --- Drag and Drop Helper ---
    function setupDragAndDrop(areaElement, callback) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            areaElement.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        areaElement.addEventListener('dragenter', () => areaElement.classList.add('dragover'), false);
        areaElement.addEventListener('dragleave', () => areaElement.classList.remove('dragover'), false);
        areaElement.addEventListener('dragover', () => areaElement.classList.add('dragover'), false); // Keep class

        areaElement.addEventListener('drop', (event) => {
            areaElement.classList.remove('dragover');
            const dt = event.dataTransfer;
            callback(dt.files);
        }, false);
    }

    // --- Update Threshold Display ---
    if (thresholdSlider && thresholdValueDisplay) {
        thresholdSlider.addEventListener('input', (event) => {
            thresholdValueDisplay.textContent = `${event.target.value}%`;
        });
    }
    // Similar listeners for weight sliders if their values need to be displayed
    document.querySelectorAll('.criteria-weights input[type="range"]').forEach(slider => {
        const valueSpan = slider.nextElementSibling; // Assuming span is right after input
        if (valueSpan && valueSpan.classList.contains('weight-value')) {
            valueSpan.textContent = `${slider.value}%`; // Initial display
            slider.addEventListener('input', () => {
                valueSpan.textContent = `${slider.value}%`;
            });
        }
    });


    // --- Screen Button Logic ---
    if (screenBtn) {
        screenBtn.addEventListener('click', async () => {
            if (screenBtn.disabled) return;

            loadingTitle.textContent = 'Screening Candidates...';
            loadingMessage.textContent = 'Analyzing resumes against the job description. This may take a moment.';
            loadingOverlay.style.display = 'flex';

            // Show popup notification for loading
            showPopupNotification('Screening Candidates...', 'Analyzing resumes against the job description. This may take a moment.', 'info');

            const threshold = parseFloat(thresholdSlider.value) / 100;
            const weights = {
                skills: parseFloat(skillsWeightSlider.value) / 100,
                experience: parseFloat(experienceWeightSlider.value) / 100,
                education: parseFloat(educationWeightSlider.value) / 100,
                // keywords: parseFloat(keywordsWeightSlider.value) / 100 // If backend supports it
            };

            // Normalize weights if they don't sum to 1 (optional, backend might handle this or expect raw values)
            // const totalWeight = Object.values(weights).reduce((sum, val) => sum + val, 0);
            // if (totalWeight > 0) {
            //     for (const key in weights) {
            //         weights[key] = weights[key] / totalWeight;
            //     }
            // }

            try {
                const response = await fetch('/screen_candidates', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold, weights })
                });
                const screeningData = await response.json();

                if (response.ok) {
                    showPopupNotification('Success', screeningData.message || 'Screening completed!', 'success');
                    displayResults(screeningData);
                    screeningResultsAvailable = true; // Set flag

                    if (screeningData.automated_notifications_summary && autoNotificationStatusDiv && autoNotificationStatusText) {
                        const summary = screeningData.automated_notifications_summary;
                        if (summary.attempted > 0) {
                            autoNotificationStatusText.textContent = `Automated notifications: ${summary.successful} sent, ${summary.failed} failed out of ${summary.attempted} shortlisted.`;
                            autoNotificationStatusDiv.style.display = 'flex';
                        } else {
                            autoNotificationStatusText.textContent = 'No automated notifications sent (no shortlisted candidates or system issue).';
                            autoNotificationStatusDiv.style.display = 'flex';
                        }
                    }
                    updateScreenButtonState(); // Update button states
                } else {
                    showPopupNotification('Error', screeningData.error || 'Screening failed.', 'error');
                    resultsSection.style.display = 'none';
                }
            } catch (error) {
                console.error('Screening API call error:', error);
                showPopupNotification('Error', 'An error occurred while contacting the server for screening.', 'error');
                resultsSection.style.display = 'none';
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // --- Display Results Function ---
    function displayResults(data) {
        resultsContent.innerHTML = ''; // Clear previous results
        if (autoNotificationStatusDiv) autoNotificationStatusDiv.style.display = 'none'; // Hide by default
        if (!data.results || data.results.length === 0) {
            resultsContent.innerHTML = '<p class="no-results">No candidates found or processed.</p>';
            resultsSection.style.display = 'block'; // Show section even if no results
            document.getElementById('totalProcessed').textContent = 0;
            document.getElementById('totalShortlisted').textContent = 0;
            document.getElementById('avgScore').textContent = '0%';
            screeningResultsAvailable = false;
            return;
        }

        let totalScore = 0;
        data.results.forEach(candidate => {
            totalScore += candidate.match_score;
            const card = document.createElement('div');
            card.className = `candidate-card ${candidate.shortlisted ? 'shortlisted' : 'rejected'}`;

            const scorePercentage = (candidate.match_score * 100).toFixed(1);
            let scoreClass = 'low';
            if (scorePercentage >= 75) scoreClass = 'high';
            else if (scorePercentage >= 50) scoreClass = 'medium';

            card.innerHTML = `
                <div class="candidate-header" style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1.2rem;">
                    <div class="candidate-info" style="flex: 1 1 0; min-width: 0;">
                        <h4 style="word-break: break-word;">${candidate.candidate} ${candidate.shortlisted
                            ? '<span class=\'status shortlisted\' style=\'margin-left:0.5rem;vertical-align:middle;\'><i class=\'fas fa-check-circle\'></i> Shortlisted</span>'
                            : '<span class=\'status rejected\' style=\'margin-left:0.5rem;vertical-align:middle;\'><i class=\'fas fa-times-circle\'></i> Rejected</span>'}
                        </h4>
                        <span class="email"><i class="fas fa-envelope"></i> ${candidate.email || 'N/A'}</span>
                        ${data.job_title ? `<span class='job-title'><i class='fas fa-briefcase'></i> Job Title: ${data.job_title}</span>` : ''}
                    </div>
                    <div class="candidate-metrics" style="display: flex; flex-direction: column; align-items: flex-end; min-width: 80px;">
                        <div class="score ${scoreClass}" style="font-size:2.1rem; font-weight:800;">${scorePercentage}%</div>
                    </div>
                </div>
                <div class="candidate-details">
                    <div style="background: #fdf6f0; border-radius: 1rem; padding: 1rem 1.2rem; margin-bottom: 0.5rem;">
                        <strong>Skill Score:</strong> ${(candidate.details.skill_score * 100).toFixed(1)}% |
                        <strong>Experience Score:</strong> ${(candidate.details.experience_score * 100).toFixed(1)}% |
                        <strong>Education Score:</strong> ${(candidate.details.education_score * 100).toFixed(1)}%
                        <br><strong>Matched Skills:</strong> ${candidate.details.matched_skills.join(', ') || 'None'}
                    </div>
                </div>
            `;
            resultsContent.appendChild(card);
        });

        // Update summary stats in results header
        document.getElementById('totalProcessed').textContent = data.total_candidates;
        document.getElementById('totalShortlisted').textContent = data.shortlisted_candidates;
        const avgScore = data.total_candidates > 0 ? (totalScore / data.total_candidates * 100).toFixed(1) : 0;
        document.getElementById('avgScore').textContent = `${avgScore}%`;

        // Update main header stats
        document.getElementById('totalCandidates').textContent = data.total_candidates;
        document.getElementById('shortlistedCount').textContent = data.shortlisted_candidates;
        // document.getElementById('processingTime').textContent = 'N/A'; // Or calculate if backend provides

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Reset Button Logic ---
    if (resetBtn) {
        resetBtn.addEventListener('click', async () => {
            // Add a confirmation modal here if desired
            try {
                const response = await fetch('/reset_system', { method: 'POST' });
                const result = await response.json();
                if (response.ok) {
                    showPopupNotification('Success', result.message || 'System reset successfully.', 'success');
                    // Clear UI elements
                    jobFilePreview.innerHTML = ''; jobFilePreview.classList.remove('show');
                    resumeFilePreview.innerHTML = ''; resumeFilePreview.classList.remove('show');
                    jobStatus.innerHTML = '<i class="fas fa-clock"></i>';
                    jobStatus.classList.remove('completed');
                    resumeStatus.innerHTML = '<i class="fas fa-clock"></i>';
                    resumeStatus.classList.remove('completed');
                    jobFileUploaded = false; resumeFilesUploaded = false;
                    updateScreenButtonState();
                    screeningResultsAvailable = false; // Reset flag
                    // Clear results display if any
                    resultsContent.innerHTML = '';
                    resultsSection.style.display = 'none';
                    document.getElementById('totalCandidates').textContent = 0;
                    document.getElementById('shortlistedCount').textContent = 0;
                    if (autoNotificationStatusDiv) autoNotificationStatusDiv.style.display = 'none';
                } else {
                    showPopupNotification('Error', result.error || 'Failed to reset system.', 'error');
                }
            } catch (error) {
                showPopupNotification('Error', 'Error resetting system.', 'error');
            }
        });
    }
    // --- Event listeners for export buttons (placeholder actions) ---
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener('click', () => {
            if(exportPdfBtn.disabled) return;
            window.location.href = '/download_results'; // This should trigger the PDF download
        });
    }

    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', () => {
            if(exportCsvBtn.disabled) return;
            // This tells the browser to navigate to the CSV download URL.
            // The backend (app.py) will send headers to make it a download.
            window.location.href = '/download_results_csv';
        });
    }

    // Initial state
    updateScreenButtonState();

    // --- SCROLL BLUR EFFECT LOGIC ---
    function updateScrollBlur() {
        const topBlur = document.querySelector('.scroll-blur-top');
        const bottomBlur = document.querySelector('.scroll-blur-bottom');
        const scrollY = window.scrollY;
        const windowHeight = window.innerHeight;
        const docHeight = document.body.scrollHeight;
        // Fade out top blur as you scroll down, fade in as you reach top
        if (topBlur) {
            topBlur.style.opacity = Math.max(0, 1 - scrollY / 120);
        }
        // Fade out bottom blur as you reach bottom, fade in as you scroll up
        if (bottomBlur) {
            const fromBottom = docHeight - (scrollY + windowHeight);
            bottomBlur.style.opacity = Math.max(0, 1 - (fromBottom / 120));
        }
    }
    window.addEventListener('scroll', updateScrollBlur);
    updateScrollBlur();
});