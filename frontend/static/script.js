const canvas = document.getElementById("waveform");
const ctx = canvas.getContext("2d");

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

function startRecording() {
    clearCanvas();

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(mediaStream => {
            stream = mediaStream;

            mediaRecorder = new MediaRecorder(mediaStream);
            recordedChunks = [];

            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });

                // Показать кнопку прослушивания
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

// Очистка
document.getElementById("clearBtn").addEventListener("click", () => {
    clearCanvas();
    recordedChunks = [];
    recordedBlob = null;
    document.getElementById("listenBtn").style.display = "none"; // Скрыть кнопку прослушивания
});

// Импорт
document.getElementById("importBtn").addEventListener("click", () => {
    document.getElementById("importFile").click();
});

document.getElementById("importFile").addEventListener("change", function () {
    const file = this.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const arrayBuffer = e.target.result;

        const context = new (window.AudioContext || window.webkitAudioContext)();
        context.decodeAudioData(arrayBuffer)
            .then(audioBuffer => {
                // Сохраняем blob для кнопки воспроизведения
                recordedBlob = file;

                // Показать кнопку прослушивания
                document.getElementById("listenBtn").style.display = "inline-block";

                // Отображаем waveform
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
