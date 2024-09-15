import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, ArrowLeft, Trash2, Upload } from 'lucide-react';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';

interface Patient {
  id: number;
  name: string;
  callTime: string;
  isRecording: boolean;
  audioBlob: Blob | null;
  audioUrl: string | null;
}

interface HomePageProps {
  patients: Patient[];
  onNewCall: () => void;
  onToggleRecording: (id: number) => void;
  onSelectPatient: (id: number) => void;
  onDeletePatient: (id: number) => void;
  showRecordingConfirmation: boolean;
  onCloseRecordingConfirmation: () => void;
}

const HomePage: React.FC<HomePageProps> = ({
  patients,
  onNewCall,
  onToggleRecording,
  onSelectPatient,
  onDeletePatient,
  showRecordingConfirmation,
  onCloseRecordingConfirmation
}) => (
  <div className="p-4">
    <h1 className="text-2xl font-bold mb-4">Call Log {new Date().toLocaleDateString()}</h1>
    <button 
      className="bg-green-500 text-white px-4 py-2 rounded mb-4"
      onClick={onNewCall}
    >
      New Call
    </button>
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
  onSendAudioToBackend: (patientId: number) => void;
}

const PatientDetailsPage: React.FC<PatientDetailsPageProps> = ({ patient, onBackToHome, onSendAudioToBackend }) => {
  if (!patient) return <div>Loading...</div>;

  return (
    <div className="p-4">
      <button 
        className="flex items-center text-blue-500 mb-4"
        onClick={onBackToHome}
      >
        <ArrowLeft className="mr-2" />
        Back to Call Log
      </button>
      <h1 className="text-2xl font-bold mb-4">{patient.name}</h1>
      <p>Call time: {patient.callTime}</p>
      <div className="my-4">
        <h2 className="font-bold">Recorded audio</h2>
        {patient.audioUrl ? (
          <div>
            <audio controls src={patient.audioUrl} className="w-full mb-2">
              Your browser does not support the audio element.
            </audio>
            <button
              className="bg-blue-500 text-white px-4 py-2 rounded flex items-center"
              onClick={() => onSendAudioToBackend(patient.id)}
            >
              <Upload size={16} className="mr-2" />
              Send to Backend
            </button>
          </div>
        ) : (
          <p>No audio recorded yet.</p>
        )}
      </div>
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
  const activePatientId = useRef<number | null>(null); // Store the active recording patient's ID

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
            // setPatients(prev => prev.map(p => { ...p, id: 99, audioBlob: audioBlob, audioUrl: audioUrl, isRecording: false }))
            
            console.log(activePatientId.current)
            var apid = activePatientId.current
            if (activePatientId.current !== null) {
              console.log('hi')
              console.log(activePatientId.current)
              console.log(apid)
              setPatients(prevPatients =>
                prevPatients.map(p =>
                  p.id === apid
                    ? { ...p, audioBlob: audioBlob, audioUrl: audioUrl, isRecording: false } // Directly update the correct patient
                    : p
                )
              );
            //   setPatients(prevPatients => prevPatients.map(p => p.id === 3 ? { ...p, audioBlob: audioBlob, audioUrl: audioUrl, isRecording: true }))
            
                // console.log(audioBlob)
                // console.log(patients)
            }

            audioChunks.current = [];
            activePatientId.current = null; // Reset after stopping
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
      audioUrl: null
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
    console.log(patients)
    if (mediaRecorder) {
      if (mediaRecorder.state === 'inactive') {
        audioChunks.current = [];
        mediaRecorder.start();
        activePatientId.current = patientId; // Store the active recording patient
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

  const sendAudioToBackend = async (patientId: number) => {
    const patient = patients.find(p => p.id === patientId);
    if (!patient || !patient.audioBlob) {
      console.error('No audio found for this patient');
      return;
    }

    const formData = new FormData();
    formData.append('audio', patient.audioBlob, `patient_${patientId}_recording.webm`);

    try {
      const response = await fetch('YOUR_BACKEND_API_ENDPOINT', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        console.log('Audio sent successfully');
      } else {
        console.error('Failed to send audio');
      }
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
        />
      )}
      {currentPage === 'patientDetails' && (
        <PatientDetailsPage 
          patient={patients.find(p => p.id === selectedPatientId) || null}
          onBackToHome={navigateToHome}
          onSendAudioToBackend={sendAudioToBackend}
        />
      )}
    </div>
  );
};

export default App;
