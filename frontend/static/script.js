const canvas = document.getElementById("waveform");
const ctx = canvas.getContext("2d");
const base_url = "http://127.0.0.1:8080";

let audioContext;
let analyser;
let dataArray;
let source;
let animationId;
let stream;

let mediaRecorder;
let recordedChunks = [];
let recordedBlob = null;
let playAudio = null;

// Обработчик для раскрытия и скрытия меню
document.getElementById("dropdownToggle").addEventListener("click", function() {
    const dropdownMenu = document.getElementById("dropdownMenu");
    dropdownMenu.classList.toggle("show");
});

// Обработчик для выбора элемента в меню
const menuItems = document.querySelectorAll(".dropdown-menu li");
menuItems.forEach(item => {
    item.addEventListener("click", function() {
        const selectedValue = this.getAttribute("data-value");
        document.getElementById("dropdownToggle").textContent = this.textContent; // Меняем текст на кнопке
        document.getElementById("dropdownMenu").classList.remove("show"); // Закрываем меню после выбора
        console.log("Выбран режим: " + selectedValue); // В дальнейшем можно использовать выбранное значение
    });
});

canvas.addEventListener("click", function (e) {
    if (!recordedBlob) return;

    // Если аудио уже проигрывается, то нужно остановить его и сбросить время
    if (playAudio) {
        playAudio.pause();
        playAudio.currentTime = 0;
    }

    // Создаем новый объект Audio и начинаем воспроизведение с самого начала
    playAudio = new Audio(URL.createObjectURL(recordedBlob));
    playAudio.play();
});

// История анализов
const historyTable = document.getElementById('history-table').querySelector('tbody');
const historyTableWrapper = document.getElementById('history-table').parentElement;
let historyRecords = [];

function addHistoryRecord(filename, duration, emotion) {
    // Проверка на дубликаты по названию файла и эмоции
    if (historyRecords.some(r => r.filename === filename && r.emotion === emotion)) return;
    historyRecords.push({ filename, duration, emotion });
    const row = document.createElement('tr');
    row.innerHTML = `<td>${filename}</td><td>${duration}</td><td>${emotion}</td>`;
    historyTable.appendChild(row);
}

function clearHistoryTable() {
    historyRecords = [];
    historyTable.innerHTML = '';
}

// Генерация уникального имени для записи
function generateRecordName() {
    const now = new Date();
    const pad = n => n.toString().padStart(2, '0');
    return `record_${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.webm`;
}

// Для записи с микрофона сохраняем имя и время
let lastRecordName = '';
let recordStartTime = null;
let recordStopTime = null;

function startRecording() {
    clearCanvas();
    lastRecordName = generateRecordName();
    recordStartTime = Date.now();
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(mediaStream => {
            stream = mediaStream;
            mediaRecorder = new MediaRecorder(mediaStream);
            recordedChunks = [];
            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };
            mediaRecorder.onstop = () => {
                recordStopTime = Date.now();
                recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });
                recordedBlob.name = lastRecordName;
                // Сохраняем длительность записи
                recordedBlob.duration = ((recordStopTime - recordStartTime) / 1000).toFixed(2);
                document.getElementById("listenBtn").style.display = "inline-block";
            };
            mediaRecorder.start();
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 1024;
            source = audioContext.createMediaStreamSource(mediaStream);
            source.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);
            drawWaveform();
        })
        .catch(err => {
            console.error("Ошибка доступа к микрофону:", err);
        });
}

function stopRecording() {
    // Остановить отрисовку волны
    if (animationId) cancelAnimationFrame(animationId);

    // Остановить микрофон
    if (stream) stream.getTracks().forEach(track => track.stop());

    // Закрыть аудиоконтекст
    if (audioContext) audioContext.close();

    // Остановить запись
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }

    // Остановить воспроизведение импортированного или записанного аудио
    if (playAudio) {
        playAudio.pause();
        playAudio.currentTime = 0;
    }
}

function drawWaveform() {
    animationId = requestAnimationFrame(drawWaveform);

    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#007bff";
    ctx.beginPath();

    const sliceWidth = canvas.width / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Импорт
let importedFileName = '';
document.getElementById("importBtn").addEventListener("click", () => {
    document.getElementById("importFile").click();
});
document.getElementById("importFile").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;
    importedFileName = file.name;
    const reader = new FileReader();
    reader.onload = function (e) {
        const arrayBuffer = e.target.result;
        const context = new (window.AudioContext || window.webkitAudioContext)();
        context.decodeAudioData(arrayBuffer)
            .then(audioBuffer => {
                recordedBlob = file;
                recordedBlob.name = importedFileName;
                document.getElementById("listenBtn").style.display = "inline-block";
                drawImportedWaveform(audioBuffer);
            })
            .catch(err => {
                console.error("Ошибка при декодировании аудио:", err);
            });
    };
    reader.readAsArrayBuffer(file);
});

function drawImportedWaveform(audioBuffer) {
    clearCanvas();

    const rawData = audioBuffer.getChannelData(0); // Используем только один канал
    const samples = 1000; // Количество точек
    const blockSize = Math.floor(rawData.length / samples);
    const filteredData = [];

    for (let i = 0; i < samples; i++) {
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(rawData[i * blockSize + j]);
        }
        filteredData.push(sum / blockSize);
    }

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#28a745";
    ctx.beginPath();

    const width = canvas.width;
    const height = canvas.height;
    const step = width / samples;
    let x = 0;

    for (let i = 0; i < samples; i++) {
        const y = height - (filteredData[i] * height);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        x += step;
    }

    ctx.stroke();
}

document.getElementById("listenBtn").addEventListener("click", () => {
    if (!recordedBlob) return;

    // Остановить предыдущее воспроизведение
    if (playAudio) {
        playAudio.pause();
        playAudio.currentTime = 0;
    }

    playAudio = new Audio(URL.createObjectURL(recordedBlob));
    playAudio.play();
});

// Открыть панель помощи
document.getElementById("helpBtn").addEventListener("click", () => {
    document.getElementById("helpPanel").style.display = "block"; // Показываем панель
});

// Закрыть панель помощи
document.getElementById("closeHelp").addEventListener("click", () => {
    document.getElementById("helpPanel").style.display = "none"; // Скрываем панель
});

// Кнопки
document.getElementById("recordBtn").addEventListener("click", startRecording);
document.getElementById("stopBtn").addEventListener("click", stopRecording);

// Функция для отображения ошибки
function showError(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

// Функция для скрытия ошибки
function hideError() {
    const errorElement = document.getElementById('error-message');
    errorElement.style.display = 'none';
}

// Функция для отправки данных на сервер
async function sendDataToServer(audioBlob, model) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('model', model);

    try {
        const response = await fetch(`${base_url}/newRecord`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Ошибка при отправке данных');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Ошибка:', error);
        throw error;
    }
}

// Модифицируем обработчик кнопки Старт
const startBtn = document.querySelector('.start-btn');
startBtn.addEventListener('click', async () => {
    hideError();
    if (!recordedBlob) {
        showError('Сначала запишите или импортируйте аудио');
        return;
    }
    const selectedModel = document.getElementById('dropdownToggle').textContent;
    if (selectedModel === 'Выбрать режим') {
        showError('Выберите модель для анализа');
        return;
    }
    try {
        const result = await sendDataToServer(recordedBlob, selectedModel);
        // Получаем имя файла и длительность
        let filename = recordedBlob.name || generateRecordName();
        let duration = '-';
        if (typeof recordedBlob.duration !== 'undefined') {
            duration = recordedBlob.duration;
        } else {
            duration = await getAudioDuration(recordedBlob);
            duration = (duration && isFinite(duration)) ? duration.toFixed(2) : '-';
        }
        addHistoryRecord(filename, duration, result.emotion || '-');
    } catch (error) {
        showError('Произошла ошибка при обработке аудио');
    }
});

// Получение длительности аудио
function getAudioDuration(blob) {
    return new Promise(resolve => {
        const audio = document.createElement('audio');
        audio.preload = 'metadata';
        audio.onloadedmetadata = function() {
            resolve(audio.duration);
        };
        audio.onerror = function() {
            resolve(0);
        };
        audio.src = URL.createObjectURL(blob);
    });
}

// Очистка истории по кнопке Очистка
const clearBtn = document.getElementById('clearBtn');
clearBtn.addEventListener('click', () => {
    clearCanvas();
    recordedChunks = [];
    recordedBlob = null;
    document.getElementById('listenBtn').style.display = 'none';
    clearHistoryTable();
});
