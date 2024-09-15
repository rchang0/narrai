import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Play, Pause, ArrowLeft, Trash2, Upload, Edit2, Check } from 'lucide-react';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';

interface Patient {
  id: number;
  name: string;
  callTime: string;
  isRecording: boolean;
  audioBlob: Blob | null;
  audioUrl: string | null;
  summary: string | null;
  transcript: string | null;
  summaryTimestamps?: { start: number; end: number; timestamp: number }[];
}

interface HomePageProps {
  patients: Patient[];
  onNewCall: () => void;
  onToggleRecording: (id: number) => void;
  onSelectPatient: (id: number) => void;
  onDeletePatient: (id: number) => void;
  showRecordingConfirmation: boolean;
  onCloseRecordingConfirmation: () => void;
  onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
}


const HomePage: React.FC<HomePageProps> = ({
  patients,
  onNewCall,
  onToggleRecording,
  onSelectPatient,
  onDeletePatient,
  showRecordingConfirmation,
  onCloseRecordingConfirmation,
  onFileUpload
}) => (
  <div className="p-4">
    <h1 className="text-2xl font-bold mb-4">Call Log {new Date().toLocaleDateString()}</h1>
    <div className="flex space-x-2 mb-4">
      <button 
        className="bg-green-500 text-white px-4 py-2 rounded"
        onClick={onNewCall}  /* This line ensures the New Call button is properly linked */
      >
        New Call
      </button>
      <label className="bg-blue-500 text-white px-4 py-2 rounded cursor-pointer flex items-center">
        <input
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={onFileUpload}
        />
        <Upload size={16} className="mr-2" />
        Upload Audio
      </label>
    </div>
    <table className="w-full">
      <thead>
        <tr>
          <th className="text-left">Time</th>
          <th className="text-left">Patient Name</th>
          <th className="text-left">Recording</th>
          <th className="text-left">Delete</th>
        </tr>
      </thead>
      <tbody>
        {patients.map((patient) => (
          <tr key={patient.id}>
            <td>{patient.callTime}</td>
            <td>
              <button
                className="text-blue-500 hover:underline"
                onClick={() => onSelectPatient(patient.id)}
              >
                {patient.name}
              </button>
            </td>
            <td>
              <button
                className={`${patient.isRecording ? 'bg-red-500' : 'bg-green-500'} text-white px-2 py-1 rounded`}
                onClick={() => onToggleRecording(patient.id)}
              >
                {patient.isRecording ? <Pause size={16} /> : <Play size={16} />}
              </button>
            </td>
            <td>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <button className="bg-red-500 text-white px-2 py-1 rounded">
                    <Trash2 size={16} />
                  </button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Are you sure you want to delete this patient?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This action cannot be undone. This will permanently delete the patient's data and any associated recordings.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={() => onDeletePatient(patient.id)}>
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </td>
          </tr>
        ))}
      </tbody>
    </table>

    <AlertDialog open={showRecordingConfirmation} onOpenChange={onCloseRecordingConfirmation}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Recording Saved</AlertDialogTitle>
          <AlertDialogDescription>
            Your recording has been successfully saved. You can find it in the patient's details page.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction onClick={onCloseRecordingConfirmation}>OK</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  </div>
);

interface PatientDetailsPageProps {
  patient: Patient | null;
  onBackToHome: () => void;
  onSendAudioToBackend: (patientId: number) => Promise<void>;
  onUpdatePatientName: (patientId: number, newName: string) => void;
}

interface PatientDetailsPageProps {
  patient: Patient | null;
  onBackToHome: () => void;
  onSendAudioToBackend: (patientId: number) => Promise<void>;
  onUpdatePatientName: (patientId: number, newName: string) => void;
}

interface PopupProps {
  x: number;
  y: number;
  onYes: () => void;
  onNo: () => void;
}

const Popup: React.FC<PopupProps> = ({ x, y, onYes, onNo }) => (
  <div 
    className="absolute bg-white border border-gray-300 rounded shadow-lg p-2"
    style={{ left: `${x}px`, top: `${y}px` }}
  >
    <p className="mb-2">Find information in audio?</p>
    <div className="flex justify-end space-x-2">
      <button 
        className="bg-blue-500 text-white px-2 py-1 rounded"
        onClick={onYes}
      >
        Yes
      </button>
      <button 
        className="bg-gray-300 text-black px-2 py-1 rounded"
        onClick={onNo}
      >
        No
      </button>
    </div>
  </div>
);


const PatientDetailsPage: React.FC<PatientDetailsPageProps> = ({ patient, onBackToHome, onSendAudioToBackend, onUpdatePatientName }) => {
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState(patient?.name || '');
  const [isLoading, setIsLoading] = useState(false);
  const [selectionIndices, setSelectionIndices] = useState<{ start: number; end: number } | null>(null);
  const [showPopup, setShowPopup] = useState(false);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const summaryRef = useRef<HTMLPreElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  if (!patient) return <div>Loading...</div>;

  // Modify the state to include words with their global indices
  const [transcriptSentences, setTranscriptSentences] = useState<{
    text: string;
    highlightIntensity: number;
    words: { word: string; index: number }[];
  }[]>([]);

  const [maxHighlightIntensity, setMaxHighlightIntensity] = useState(0);

  // Function to split transcript into sentences
  function splitTranscriptIntoSentences(text: string): string[] {
    const regex = /[^.!?]+[.!?]+[\])'"`’”]*|.+/g;
    return text.match(regex) || [];
  }

  // Update transcript sentences when patient.transcript changes
  useEffect(() => {
    if (patient.transcript) {
      const sentences = [];
      let globalWordIndex = 0;
      const rawSentences = splitTranscriptIntoSentences(patient.transcript);
      for (const sentence of rawSentences) {
        const words = sentence.split(/\s+/).filter(i => i).map(word => ({
          word,
          index: globalWordIndex++,
        }));
        sentences.push({
          text: sentence,
          highlightIntensity: 0,
          words: words,
        });
      }
      setTranscriptSentences(sentences);
    }
  }, [patient.transcript]);
  // Function to handle word clicks
  const handleWordClick = async (wordIndex: number) => {
    const timestamp = await getTimestampFromBackend(wordIndex);
    if (timestamp >= 0 && audioRef.current) {
      audioRef.current.currentTime = timestamp;
      audioRef.current.play();
    }
  };

  // Function to get timestamp from the backend
  const getTimestampFromBackend = async (wordIndex: number): Promise<number> => {
    try {
      const response = await fetch(
        'https://frankxwang--example-whisper-streaming-testing-get-timestamp.modal.run?' +
          new URLSearchParams({
            filename: `patient_${patient.id}`,
            word_idx: String(wordIndex),
          }).toString()
      );

      if (!response.ok) {
        throw new Error('Failed to fetch timestamp from backend');
      }

      const data = await response.json();
      return Math.max(data.timestamp - 1, 0);
    } catch (error) {
      console.error('Error fetching timestamp:', error);
      return -1;
    }
  };

  const handleNameEdit = () => {
    if (isEditingName) {
      onUpdatePatientName(patient.id, editedName);
    }
    setIsEditingName(!isEditingName);
  };

  const handleSendAudio = async () => {
    setIsLoading(true);
    await onSendAudioToBackend(patient.id);
    setIsLoading(false);
  };

  const processSummaryText = (text: string) => {
    return text
      .replace(/\\n/g, '\n')
      .replace(/\\"/g, '"')
      .replace(/\\'/g, "'")
      .replace(/\\t/g, '\t')
      .replace(/\\b/g, '\b')
      .replace(/\\f/g, '\f')
      .replace(/\\r/g, '\r')
      .replace(/\\([0-7]{1,3})/g, (match, oct) =>
        String.fromCharCode(parseInt(oct, 8))
      )
      .replace(/\\x([0-9A-Fa-f]{2})/g, (match, hex) =>
        String.fromCharCode(parseInt(hex, 16))
      )
      .replace(/\\u([0-9A-Fa-f]{4})/g, (match, hex) =>
        String.fromCharCode(parseInt(hex, 16))
      );
  };

  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    const summaryElement = summaryRef.current;

    console.log("HAPPENING")
    if (selection && !selection.isCollapsed && summaryElement) {
      const range = selection.getRangeAt(0);
      const preContent = summaryElement.textContent || '';
      const start = preContent.indexOf(range.toString());
      const end = start + range.toString().length;

      if (start !== -1) {
        setSelectionIndices({ start, end });
        console.log(start)
        console.log(end)

        // POST SELECTION INDICES
        // Calculate position for the popup
        const rect = range.getBoundingClientRect();
        setPopupPosition({
          x: rect.left + window.scrollX,
          y: rect.bottom + window.scrollY
        });
        setShowPopup(true);
      }
    } else {
      setShowPopup(false);
    }
  }, []);

  const getCitationsFromBackend = async (start: number, end: number): Promise<any> => {
    try {
      const response = await fetch('https://frankxwang--example-whisper-streaming-testing-cite.modal.run?' + new URLSearchParams({
          filename: `patient_${patient.id}`,
          start_idx: String(start),
          end_idx: String(end),
      }).toString());

      if (!response.ok) {
        throw new Error('Failed to fetch timestamp from backend');
      }

      const data = await response.json();
      console.log(data);
      return data;
    } catch (error) {
      console.error('Error fetching timestamp:', error);
      return -1;
    }
  };

  // Modify handlePopupYes to update highlight intensities
  const handlePopupYes = async () => {
    if (selectionIndices) {
      const data = await getCitationsFromBackend(selectionIndices.start, selectionIndices.end);
      const timestamp = data.times[0];

      if (timestamp >= 0 && audioRef.current) {
        audioRef.current.currentTime = timestamp;
        audioRef.current.play();
      } else {
        console.log("Failed to get a valid timestamp or audio element not available");
      }

      if (data.text && Array.isArray(data.text)) {
        const maxIntensity = data.text.length;
        setMaxHighlightIntensity(maxIntensity);

        const normalizeSentence = (sentence: string) =>
          sentence.trim().replace(/[^\w\s]|_/g, "").replace(/\s+/g, " ").toLowerCase();

        setTranscriptSentences(prevSentences => {
          return prevSentences.map(sentenceObj => {
            const indexInDataText = data.text.findIndex(
              highlightSentence => normalizeSentence(highlightSentence) === normalizeSentence(sentenceObj.text)
            );
            if (indexInDataText !== -1) {
              return {
                ...sentenceObj,
                highlightIntensity: maxIntensity - indexInDataText,
              };
            } else {
              return {
                ...sentenceObj,
                highlightIntensity: 0,
              };
            }
          });
        });
      }
    }
    setShowPopup(false);
  };

  const handlePopupNo = () => {
    setShowPopup(false);
  };

  const addPopup = () => {
    console.log("POPUP ADDING")
    const summaryElement = summaryRef.current;
    if (summaryElement) {
      console.log("POPUP ADDED")
      summaryElement.addEventListener('mouseup', handleTextSelection);
      return () => {
        summaryElement.removeEventListener('mouseup', handleTextSelection);
      };
    }
  };

  useEffect(() => {
    addPopup()
  }, [handleTextSelection, patient.summary]);

  return (
    <div className="p-4 relative">
      <button 
        className="flex items-center text-blue-500 mb-4"
        onClick={onBackToHome}
      >
        <ArrowLeft className="mr-2" />
        Back to Call Log
      </button>
      <div className="flex items-center mb-4">
        {isEditingName ? (
          <input
            type="text"
            value={editedName}
            onChange={(e) => setEditedName(e.target.value)}
            className="text-2xl font-bold mr-2 border-b-2 border-blue-500 focus:outline-none"
          />
        ) : (
          <h1 className="text-2xl font-bold mr-2">{patient.name}</h1>
        )}
        <button onClick={handleNameEdit} className="text-blue-500">
          {isEditingName ? <Check size={20} /> : <Edit2 size={20} />}
        </button>
      </div>
      <p>Call time: {patient.callTime}</p>
      <div className="my-4">
        <h2 className="font-bold">Recorded audio</h2>
        {patient.audioUrl ? (
          <div>
            <audio ref={audioRef} controls src={patient.audioUrl} className="w-full mb-2">
              Your browser does not support the audio element.
            </audio>
            <button
              className={`bg-blue-500 text-white px-4 py-2 rounded flex items-center ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={handleSendAudio}
              disabled={isLoading}
            >
              <Upload size={16} className="mr-2" />
              {isLoading ? 'Processing...' : 'Create Narrative Outline'}
            </button>
          </div>
        ) : (
          <p>No audio recorded yet.</p>
        )}
      </div>
      {patient.summary && (
        <div className="my-4">
          <h2 className="font-bold">Summary</h2>
          <pre 
            ref={summaryRef}
            className="whitespace-pre-wrap font-sans"
          >
            {processSummaryText(patient.summary)}
          </pre>
        </div>
      )}
      
      {patient.transcript && (
        <div className="my-4">
          <h2 className="font-bold">Transcript</h2>
          {transcriptSentences.length > 0 ? (
            <div className="whitespace-pre-wrap">
              {transcriptSentences.map((sentenceObj, sentenceIndex) => (
                <span
                  key={sentenceIndex}
                  style={{
                    backgroundColor:
                      sentenceObj.highlightIntensity > 0
                        ? `rgba(255, 255, 0, ${sentenceObj.highlightIntensity / maxHighlightIntensity})`
                        : 'transparent',
                  }}
                >
                  {sentenceObj.words.map((wordObj) => (
                    <span
                      key={wordObj.index}
                      onClick={() => handleWordClick(wordObj.index)}
                      style={{ cursor: 'pointer', color: 'blue' }}
                    >
                      {wordObj.word}{' '}
                    </span>
                  ))}
                </span>
              ))}
            </div>
          ) : (
            <p className="whitespace-pre-wrap">
              {patient.transcript && (patient.transcript.split(/\s+/).filter(i => i).map((word, index) => (
                <span
                  key={index}
                  onClick={() => handleWordClick(index)}
                  style={{ cursor: 'pointer', color: 'blue' }}
                >
                  {word}{' '}
                </span>
              )))}
            </p>
          )}
        </div>
      )}

      {showPopup && (
        <Popup
          x={popupPosition.x}
          y={popupPosition.y}
          onYes={handlePopupYes}
          onNo={handlePopupNo}
        />
      )}
    </div>
  );
};


const App: React.FC = () => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [currentPage, setCurrentPage] = useState<'home' | 'patientDetails'>('home');
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [showRecordingConfirmation, setShowRecordingConfirmation] = useState(false);
  const audioChunks = useRef<BlobPart[]>([]);
  const nextPatientId = useRef<number>(1);
  const activePatientId = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const updatePatientName = (patientId: number, newName: string) => {
    setPatients(prevPatients =>
      prevPatients.map(p =>
        p.id === patientId ? { ...p, name: newName } : p
      )
    );
  };

  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
          recorder.ondataavailable = (e) => {
            audioChunks.current.push(e.data);
          };
          recorder.onstop = () => {
            const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            var apid = activePatientId.current
            if (apid !== null) {
              setPatients(prevPatients =>
                prevPatients.map(p =>
                  p.id === apid
                    ? {
                      ...p,
                      audioBlob: p.audioBlob
                        ? new Blob([p.audioBlob, audioBlob], { type: 'audio/webm' })
                        : audioBlob,
                      audioUrl: p.audioBlob
                        ? URL.createObjectURL(
                            new Blob([p.audioBlob, audioBlob], { type: 'audio/webm' })
                          )
                        : audioUrl,
                      isRecording: false
                    }
                    : p
                )
              );
            }

            audioChunks.current = [];
            activePatientId.current = null;
            setShowRecordingConfirmation(true);
          };
          setMediaRecorder(recorder);
        })
        .catch(err => console.error("Error accessing the microphone", err));
    } else {
      console.error("getUserMedia is not supported in this browser");
    }
  }, []);

  const startNewCall = () => {
    const newPatient: Patient = {
      id: nextPatientId.current,
      name: `Patient ${nextPatientId.current} Name`,
      callTime: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isRecording: false,
      audioBlob: null,
      audioUrl: null,
      summary: null,
      transcript: null
    };
    setPatients(prevPatients => [...prevPatients, newPatient]);
    nextPatientId.current += 1;
  };

  const toggleRecording = (patientId: number) => {
    setPatients(prevPatients => 
      prevPatients.map(patient => 
        patient.id === patientId
          ? { ...patient, isRecording: !patient.isRecording }
          : { ...patient, isRecording: false }
      )
    );
    if (mediaRecorder) {
      if (mediaRecorder.state === 'inactive') {
        audioChunks.current = [];
        mediaRecorder.start();
        activePatientId.current = patientId;
      } else {
        mediaRecorder.stop();
      }
    }
  };

  const deletePatient = (patientId: number) => {
    setPatients(prevPatients => {
      const updatedPatients = prevPatients.filter(patient => patient.id !== patientId);
      const deletedPatient = prevPatients.find(p => p.id === patientId);
      if (deletedPatient && deletedPatient.audioUrl) {
        URL.revokeObjectURL(deletedPatient.audioUrl);
      }
      return updatedPatients;
    });
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const newPatient: Patient = {
        id: nextPatientId.current,
        name: `Patient ${nextPatientId.current} (Uploaded)`,
        callTime: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isRecording: false,
        audioBlob: file,
        audioUrl: URL.createObjectURL(file),
        summary: null,
        transcript: null
      };
      setPatients(prevPatients => [...prevPatients, newPatient]);
      nextPatientId.current += 1;
    }
  };

  const poll = async (fn: any, callId: string, interval: any) => {
    let result;
  
    // Continue polling until you get the desired result
    while (!result) {
      result = await fn(callId);
      if (result) {
        return result;
      }
      await new Promise(resolve => setTimeout(resolve, interval)); // Wait for the next poll
    }
  };
  
  // Example function that checks a condition
  const fetchData = async (callId: string) => {
    const response = await fetch(`https://frankxwang--example-whisper-streaming-testing-web.modal.run/result/${callId}`);
    // const data = await response.json();
  
    // You can define your condition for stopping polling
    if (response.status === 202) {
      return null;
    } else {
      return await response.json(); // Continue polling if condition isn't met
    }
  };
  
  // Usage
  const runPolling = async (callId: string) => {
    const data = await poll(fetchData, callId, 5000); // Poll every 5 seconds
    console.log('Data received:', data);
    return data
  };

  const sendAudioToBackend = async (patientId: number) => {
    const patient = patients.find(p => p.id === patientId);
    if (!patient || !patient.audioBlob) {
      console.error('No audio found for this patient');
      return;
    }

    const formData = new FormData();
    formData.append('audio', patient.audioBlob, `patient_${patientId}`);

    try {
      const response = await fetch('https://frankxwang--example-whisper-streaming-testing-web.modal.run/transcribe', {
        method: 'POST',
        body: formData,
        redirect: 'follow'
      });
      
      const call_id = await response.json()
      const resultData = await runPolling(call_id.call_id);

      setPatients(prevPatients =>
        prevPatients.map(p =>
          p.id === patientId
            ? { ...p, summary: resultData.raw_llama_output, transcript: resultData.transcript }
            : p
        )
      );
    } catch (error) {
      console.error('Error sending audio:', error);
    }
  };

  const navigateToHome = () => {
    setCurrentPage('home');
    setSelectedPatientId(null);
  };

  const navigateToPatientDetails = (patientId: number) => {
    setSelectedPatientId(patientId);
    setCurrentPage('patientDetails');
  };

  const closeRecordingConfirmation = () => {
    setShowRecordingConfirmation(false);
  };

  return (
    <div className="container mx-auto">
      {currentPage === 'home' && (
        <HomePage 
          patients={patients}
          onNewCall={startNewCall}
          onToggleRecording={toggleRecording}
          onSelectPatient={navigateToPatientDetails}
          onDeletePatient={deletePatient}
          showRecordingConfirmation={showRecordingConfirmation}
          onCloseRecordingConfirmation={closeRecordingConfirmation}
          onFileUpload={handleFileUpload}
        />
      )}
      {currentPage === 'patientDetails' && (
        <PatientDetailsPage 
          patient={patients.find(p => p.id === selectedPatientId) || null}
          onBackToHome={navigateToHome}
          onSendAudioToBackend={sendAudioToBackend}
          onUpdatePatientName={updatePatientName}
        />
      )}
    </div>
  );
};

export default App;