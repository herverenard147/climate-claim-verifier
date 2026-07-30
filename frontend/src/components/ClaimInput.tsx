import React, { useState, useEffect } from 'react';
import { Search, Loader2, Mic, MicOff } from 'lucide-react';

interface ClaimInputProps {
  claim: string;
  setClaim: (val: string) => void;
  onVerify: (text: string) => void;
  isLoading: boolean;
}

// Déclaration pour TypeScript (Web Speech API)
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

export default function ClaimInput({ claim, setClaim, onVerify, isLoading }: ClaimInputProps) {
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState<any>(null);

  useEffect(() => {
    // Initialisation de l'API de reconnaissance vocale native du navigateur
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const rec = new SpeechRecognition();
        rec.continuous = false;
        rec.interimResults = true;
        rec.lang = 'fr-FR';

        rec.onresult = (event: any) => {
          let currentTranscript = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            currentTranscript += event.results[i][0].transcript;
          }
          setClaim(currentTranscript);
        };

        rec.onerror = (event: any) => {
          console.error("Erreur micro:", event.error);
          setIsListening(false);
        };

        rec.onend = () => {
          setIsListening(false);
        };

        setRecognition(rec);
      }
    }
  }, [setClaim]);

  const toggleListen = () => {
    if (isListening) {
      recognition?.stop();
      setIsListening(false);
    } else {
      if (recognition) {
        setClaim(""); // Vide le champ avant de commencer à dicter
        recognition.start();
        setIsListening(true);
      } else {
        alert("La reconnaissance vocale n'est pas supportée sur ce navigateur.");
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onVerify(claim);
    }
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#E2E8F0] relative focus-within:ring-2 focus-within:ring-[#059669]/20 focus-within:border-[#059669] transition-all">
      <textarea
        value={claim}
        onChange={(e) => setClaim(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder='Ex. : « Le niveau des océans a augmenté de 20 cm depuis 1900 » — une affirmation, pas une question. Collez, saisissez ou dictez. (Entrée pour vérifier)'
        className="w-full resize-none outline-none text-lg text-[#334155] placeholder:text-[#94A3B8] min-h-[120px] bg-transparent"
      />
      
      <div className="absolute bottom-6 left-6">
        <button
          onClick={toggleListen}
          title={isListening ? "Arrêter l'enregistrement" : "Dicter une note vocale"}
          className={`flex items-center justify-center w-12 h-12 rounded-full transition-all shadow-md ${
            isListening 
              ? "bg-red-100 text-red-600 hover:bg-red-200 animate-pulse" 
              : "bg-[#F1F5F9] text-[#64748B] hover:bg-[#E2E8F0] hover:text-[#059669]"
          }`}
        >
          {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
        </button>
      </div>

      <div className="absolute bottom-6 right-6">
        <button 
          onClick={() => onVerify(claim)}
          disabled={isLoading || !claim.trim()}
          className="flex items-center gap-2 bg-[#059669] hover:bg-[#047857] disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-xl shadow-md hover:shadow-lg transition-all"
        >
          {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
          Vérifier la déclaration
        </button>
      </div>
    </div>
  );
}
