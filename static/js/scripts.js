let recognizing = false;
let audioChunks = [];
let recognition;


document.getElementById('recordBtn').addEventListener('click', function() {
    if(recognizing){
        recognition.stop();
        return;
    }
    
    this.disabled = true;
    this.textContent = 'Initializing...';
    document.getElementById('status').textContent = 'Initializing...';
    
    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = document.getElementById('languageSelect').value;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        recognizing = true;
        this.textContent = 'Stop Recording';
        document.getElementById('status').textContent = 'Recording... Speak now.';
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById('transcription').value = transcript;
        generateNote(transcript);
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        document.getElementById('status').textContent = `Error: ${event.error}`;
        this.disabled = false;
        this.textContent = 'Start Recording';
    };

    recognition.onend = () => {
        recognizing = false;
        this.disabled = false;
        this.textContent = 'Start Recording';
        document.getElementById('status').textContent = 'Recording stopped.';
    };

    recognition.start();
});
      
function generateNote(transcription) {
    document.getElementById('status').textContent = 'Generating note...';

    fetch('/transcribe', {
        method: 'POST',
        body: new URLSearchParams({
            'transcription': transcription
        })
    })
    .then(response => response.json())
    .then(data => {
        if(data.error){
            document.getElementById('status').textContent = data.error;
        } else {
            document.getElementById('note').value = data.note;
            document.getElementById('status').textContent = 'Note generated successfully.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('status').textContent = 'An error occurred.';
    });
}