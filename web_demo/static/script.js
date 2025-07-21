document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const processBtn = document.getElementById('process-btn');

    const uploadSection = document.getElementById('upload-section');
    const statusSection = document.getElementById('status-section');
    const resultsSection = document.getElementById('results-section');

    const statusMessage = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const stepName = document.getElementById('step-name');
    const resultsDisplay = document.getElementById('results-display');
    const retentionRateEl = document.getElementById('retention-rate');
    const downloadBtn = document.getElementById('download-btn');
    
    let uploadedFiles = [];
    let currentTaskId = null;

    // --- Drag and Drop ---
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
    });

    function handleFiles(files) {
        let totalSize = uploadedFiles.reduce((acc, file) => acc + file.size, 0);
        for (const file of files) {
            totalSize += file.size;
        }

        if (totalSize > 1 * 1024 * 1024 * 1024) {
            alert('文件总大小不能超过 1GB。');
            return;
        }

        for (const file of files) {
            uploadedFiles.push(file);
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            fileList.appendChild(fileItem);
        }

        if (uploadedFiles.length > 0) {
            processBtn.disabled = false;
        }
    }

    // --- API Calls ---
    processBtn.addEventListener('click', async () => {
        if (uploadedFiles.length === 0) return;

        processBtn.disabled = true;
        processBtn.textContent = '正在上传...';

        const formData = new FormData();
        uploadedFiles.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '上传失败。');
            }
            
            currentTaskId = data.task_id;
            startProcessing(currentTaskId);

        } catch (error) {
            alert(`错误: ${error.message}`);
            resetUI();
        }
    });

    async function startProcessing(taskId) {
        uploadSection.style.display = 'none';
        statusSection.style.display = 'block';
        statusMessage.textContent = '正在初始化处理任务...';

        try {
            const response = await fetch(`/process/${taskId}`, { method: 'POST' });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || '无法启动处理流程。');
            }
            
            pollStatus(taskId);

        } catch (error) {
            statusMessage.textContent = `错误: ${error.message}`;
        }
    }

    function pollStatus(taskId) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/status/${taskId}`);
                const data = await response.json();

                // Update status message, progress bar, and step name
                statusMessage.textContent = data.message || '正在获取状态...';
                if (data.progress !== undefined) {
                    const progress = Math.min(data.progress, 100);
                    progressBar.style.width = `${progress}%`;
                    progressText.textContent = `${progress}%`;
                }
                if (data.step) {
                    stepName.textContent = data.step;
                }


                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(interval);
                    handleCompletion(data);
                }

            } catch (error) {
                clearInterval(interval);
                statusMessage.textContent = '检查状态时发生错误。';
            }
        }, 2000); // Poll every 2 seconds for smoother updates
    }
    
    function handleCompletion(data) {
        statusSection.style.display = 'none';
        resultsSection.style.display = 'block';

        if (data.status === 'failed') {
            resultsDisplay.innerHTML = `<p style="color:var(--error-color);">处理失败: ${data.message}</p>`;
            if (data.data && data.data.log) {
                resultsDisplay.innerHTML += `<h4>错误日志:</h4><pre>${data.data.log}</pre>`;
            }
            return;
        }

        // Populate results
        const { results, retention_rate } = data.data;
        retentionRateEl.textContent = retention_rate || "未能计算";

        if (!results || results.length === 0) {
            resultsDisplay.innerHTML = '<p>处理完成，但未发现有效的语音片段。😥</p>';
            downloadBtn.style.display = 'none';
        } else {
            results.forEach(speaker => {
                const speakerGroup = document.createElement('div');
                speakerGroup.className = 'speaker-group';
                
                const speakerTitle = document.createElement('h3');
                speakerTitle.textContent = `说话人: ${speaker.speaker_id}`;
                speakerGroup.appendChild(speakerTitle);

                speaker.segments.forEach(seg => {
                    const item = document.createElement('div');
                    item.className = 'result-item';
                    item.innerHTML = `
                        <p>"${seg.text}"</p>
                        <audio controls src="${seg.audio_url}"></audio>
                    `;
                    speakerGroup.appendChild(item);
                });
                resultsDisplay.appendChild(speakerGroup);
            });

            downloadBtn.onclick = () => {
                window.location.href = `/download/${currentTaskId}`;
            };
        }
    }

    function resetUI() {
        processBtn.disabled = true;
        processBtn.textContent = '开始处理';
        fileList.innerHTML = '';
        uploadedFiles = [];
        uploadSection.style.display = 'block';
        statusSection.style.display = 'none';
        resultsSection.style.display = 'none';
    }
});
