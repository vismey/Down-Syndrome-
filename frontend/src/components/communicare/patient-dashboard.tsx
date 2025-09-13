"use client";

import { useState, useEffect, useRef, type ElementType } from "react";
import {
  Mic,
  Volume2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

// Extend Window interface for Speech Recognition API
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

interface ActionButton {
  label: string;
  fullText: string;
  emoji: string;
}

const actionButtons: ActionButton[] = [
  { label: "Hungry", fullText: "I am hungry", emoji: "🍎" },
  { label: "Thirsty", fullText: "I am thirsty", emoji: "💧" },
  { label: "Medicine", fullText: "I need my medicine", emoji: "💊" },
  { label: "Help", fullText: "I need help", emoji: "🙋" },
  { label: "Pain", fullText: "I am in pain", emoji: "🤕" },
  { label: "Restroom", fullText: "I need to use the restroom", emoji: "🚽" },
];

export default function PatientDashboard() {
  const [message, setMessage] = useState<string>("");
  const [isListening, setIsListening] = useState<boolean>(false);
  const recognitionRef = useRef<any>(null);

  const speak = (text: string) => {
    if (typeof window !== "undefined" && window.speechSynthesis && text) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
    }
  };
  
  useEffect(() => {
    if (typeof window === "undefined") return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("Speech recognition not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    
    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      const speechResult = event.results[0][0].transcript;
      setMessage(speechResult);
      speak(speechResult);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
    };
    
    recognitionRef.current = recognition;

  }, []);

  const handleActionClick = (fullText: string) => {
    setMessage(fullText);
    speak(fullText);
  };

  const toggleListen = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      try {
        recognitionRef.current.start();
      } catch (error) {
        console.error("Could not start recognition:", error);
      }
    }
  };

  return (
    <div className="flex flex-col items-center gap-8">
      <Card className="w-full max-w-4xl shadow-lg">
        <CardContent className="p-6">
          <div className="flex min-h-[100px] items-center justify-between">
            <p className="flex-grow text-center font-headline text-2xl md:text-4xl">
              {message || ""}
            </p>
            {message && (
              <Button variant="ghost" size="icon" onClick={() => speak(message)} aria-label="Speak message">
                <Volume2 className="h-8 w-8 text-accent" />
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid w-full max-w-4xl grid-cols-2 gap-4 md:grid-cols-3 md:gap-6">
        {actionButtons.map(({ label, fullText, emoji }) => (
          <Button
            key={label}
            onClick={() => handleActionClick(fullText)}
            className="flex h-32 w-full transform flex-col items-center justify-center rounded-2xl text-2xl font-bold shadow-md transition-transform hover:scale-105 md:h-40 bg-secondary text-secondary-foreground hover:bg-secondary/80"
          >
            <span className="text-6xl mb-2">{emoji}</span>
            <span className="text-center text-xl">{label}</span>
          </Button>
        ))}
      </div>

      <div className="flex w-full max-w-4xl gap-4">
        <Button
            onClick={toggleListen}
            className={cn(
                "h-20 flex-1 rounded-2xl text-xl shadow-md",
                isListening ? "bg-red-500 hover:bg-red-600" : "bg-primary hover:bg-primary/90"
            )}
            aria-label={isListening ? 'Stop listening' : 'Start capturing'}
        >
            <Mic className="mr-4 h-8 w-8" />
            {isListening ? 'Listening...' : 'Capture'}
        </Button>
      </div>
    </div>
  );
}
