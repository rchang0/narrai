import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, ArrowLeft, Trash2 } from 'lucide-react';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';

interface Patient {
  id: number;
  name: string;
  callTime: string;
  isRecording: boolean;
  audioBlob: Blob | null;
}

// ... (keep other interfaces and components as they are)

const App: React.FC = () => {
  const [patients, setPatients] = useState<Patient[]>(initialPatients);
  const [currentPage, setCurrentPage] = useState<'home' | 'patientDetails'>('home');
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const audioChunks = useRef<BlobPart[]>([]);
  const nextPatientId = useRef<number>(Math.max(...initialPatients.map(p => p.id), 0) + 1);

  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          const recorder = new MediaRecorder(stream);
          recorder.ondataavailable = (e) => {
            audioChunks.current.push(e.data);
          };
          recorder.onstop = () => {
            const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
            setPatients(prevPatients => 
              prevPatients.map(p => 
                p.isRecording ? { ...p, audioBlob, isRecording: false } : p
              )
            );
            sendAudioToBackend(audioBlob);
            audioChunks.current = [];
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
      audioBlob: null
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
      } else {
        mediaRecorder.stop();
      }
    }
  };

  const deletePatient = (patientId: number) => {
    setPatients(prevPatients => prevPatients.filter(patient => patient.id !== patientId));
  };

  const sendAudioToBackend = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
      const response = await fetch('YOUR_BACKEND_API_ENDPOINT', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('Audio sent successfully');
        // Handle the response from your backend here
      } else {
        console.error('Failed to send audio');
      }
    } catch (error) {
      console.error('Error sending audio:', error);
    }
  };

  // ... (keep other functions like navigateToHome, navigateToPatientDetails, etc.)

  return (
    <div className="container mx-auto">
      {currentPage === 'home' && (
        <HomePage 
          patients={patients}
          onNewCall={startNewCall}
          onToggleRecording={toggleRecording}
          onSelectPatient={navigateToPatientDetails}
          onDeletePatient={deletePatient}
        />
      )}
      {currentPage === 'patientDetails' && (
        <PatientDetailsPage 
          patient={patients.find(p => p.id === selectedPatientId) || null}
          onBackToHome={navigateToHome}
        />
      )}
    </div>
  );
};

export default App;