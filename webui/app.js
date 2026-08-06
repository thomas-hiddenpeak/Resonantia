// Resonantia WebUI - Frontend Application

const API_BASE = '/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const btnRemove = document.getElementById('btnRemove');
const btnConvert = document.getElementById('btnConvert');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressInfo = document.getElementById('progressInfo');
const resultSection = document.getElementById('resultSection');
const resultAudio = document.getElementById('resultAudio');
const btnDownload = document.getElementById('btnDownload');
const btnCompare = document.getElementById('btnCompare');
const compareSection = document.getElementById('compareSection');
const originalAudio = document.getElementById('originalAudio');
const convertedAudio = document.getElementById('convertedAudio');

// State
let currentFile = null;
let originalAudioUrl = null;
let convertedAudioUrl = null;

// Parameter controls
const speakerId = document.getElementById('speakerId');
const pitchShift = document.getElementById('pitchShift');
const pitchShiftValue = document.getElementById('pitchShiftValue');
const indexRate = document.getElementById('indexRate');
const indexRateValue = document.getElementById('indexRateValue');
const rmsMixRate = document.getElementById('rmsMixRate');
const rmsMixRateValue = document.getElementById('rmsMixRateValue');
const protect = document.getElementById('protect');
const protectValue = document.getElementById('protectValue');

// Upload handlers
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border-color)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.name.match(/\.(wav|flac)$/i)) {
        alert('请上传 WAV 或 FLAC 格式的音频文件');
        return;
    }
    
    currentFile = file;
    
    // Show file info
    fileInfo.querySelector('.file-name').textContent = file.name;
    fileInfo.style.display = 'flex';
    uploadArea.style.display = 'none';
    
    // Create preview URL
    if (originalAudioUrl) {
        URL.revokeObjectURL(originalAudioUrl);
    }
    originalAudioUrl = URL.createObjectURL(file);
    originalAudio.src = originalAudioUrl;
    
    // Enable convert button
    btnConvert.disabled = false;
}

btnRemove.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    uploadArea.style.display = 'block';
    btnConvert.disabled = true;
    
    if (originalAudioUrl) {
        URL.revokeObjectURL(originalAudioUrl);
        originalAudioUrl = null;
    }
});

// Parameter value displays
pitchShift.addEventListener('input', () => {
    pitchShiftValue.textContent = pitchShift.value;
});

indexRate.addEventListener('input', () => {
    indexRateValue.textContent = (indexRate.value / 100).toFixed(2);
});

rmsMixRate.addEventListener('input', () => {
    rmsMixRateValue.textContent = (rmsMixRate.value / 100).toFixed(2);
});

protect.addEventListener('input', () => {
    protectValue.textContent = (protect.value / 100).toFixed(2);
});

// Convert handler
btnConvert.addEventListener('click', async () => {
    if (!currentFile) return;
    
    btnConvert.disabled = true;
    progressSection.style.display = 'block';
    resultSection.style.display = 'none';
    compareSection.style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('audio', currentFile);
        formData.append('speaker_id', speakerId.value);
        formData.append('f0_up_key', pitchShift.value);
        formData.append('index_rate', indexRate.value / 100);
        formData.append('rms_mix_rate', rmsMixRate.value / 100);
        formData.append('protect', protect.value / 100);
        
        // Simulate progress (backend doesn't support streaming yet)
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            
            progressFill.style.width = progress + '%';
            progressInfo.textContent = `转换中... ${Math.round(progress)}%`;
        }, 500);
        
        const response = await fetch(`${API_BASE}/convert`, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const blob = await response.blob();
        
        if (convertedAudioUrl) {
            URL.revokeObjectURL(convertedAudioUrl);
        }
        convertedAudioUrl = URL.createObjectURL(blob);
        
        // Update UI
        progressFill.style.width = '100%';
        progressInfo.textContent = '转换完成!';
        
        setTimeout(() => {
            progressSection.style.display = 'none';
            resultSection.style.display = 'block';
            
            resultAudio.src = convertedAudioUrl;
            convertedAudio.src = convertedAudioUrl;
            
            // Update download button
            btnDownload.onclick = () => {
                const a = document.createElement('a');
                a.href = convertedAudioUrl;
                a.download = 'converted_' + currentFile.name.replace(/\.[^.]+$/, '.wav');
                a.click();
            };
            
            btnConvert.disabled = false;
        }, 1000);
        
    } catch (error) {
        console.error('Conversion failed:', error);
        progressInfo.textContent = `转换失败: ${error.message}`;
        btnConvert.disabled = false;
    }
});

// Compare handler
btnCompare.addEventListener('click', () => {
    compareSection.style.display = 'block';
    
    // Sync playback
    const syncPlayback = () => {
        convertedAudio.currentTime = originalAudio.currentTime;
    };
    
    originalAudio.addEventListener('seeking', syncPlayback);
    originalAudio.addEventListener('play', () => convertedAudio.play());
    originalAudio.addEventListener('pause', () => convertedAudio.pause());
});
