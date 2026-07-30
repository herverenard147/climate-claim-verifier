import React from 'react';
import { HelpCircle, ArrowRight, Layers } from 'lucide-react';

interface Guidance {
  guidance_type: 'multiple' | 'vague' | 'incoherent';
  reason: string | null;
  segments: string[];
  message: string;
}

// Le backend ne comprend jamais le sens du texte (détection par motifs de
// surface, voir input_heuristics.py) : ce composant ne pose donc jamais de
// question ouverte, seulement des choix fermés (segments détectés, ou
// "envoyer tel quel") - le système ne peut pas tenir une conversation.
// Ne bloque jamais : "Envoyer tel quel quand même" reste toujours visible.
export default function GuidanceCard({
  guidance,
  onVerifySegment,
  onVerifyAllSeparately,
  onSendAnyway,
}: {
  guidance: Guidance;
  onVerifySegment: (segment: string) => void;
  onVerifyAllSeparately: (segments: string[]) => void;
  onSendAnyway: () => void;
}) {
  const isMultiple = guidance.guidance_type === 'multiple';

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 shadow-sm">
      <div className="flex items-start gap-3 mb-4">
        <HelpCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <p className="text-amber-900 font-medium leading-relaxed">{guidance.message}</p>
      </div>

      {isMultiple && (
        <div className="space-y-2 mb-4">
          {guidance.segments.map((segment, idx) => (
            <button
              key={idx}
              onClick={() => onVerifySegment(segment)}
              className="w-full text-left flex items-center justify-between gap-3 bg-white hover:bg-amber-100 border border-amber-200 rounded-xl px-4 py-3 transition-all"
            >
              <span className="text-sm text-[#334155]">« {segment} »</span>
              <ArrowRight className="w-4 h-4 text-amber-600 flex-shrink-0" />
            </button>
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-4">
        {isMultiple && (
          <button
            onClick={() => onVerifyAllSeparately(guidance.segments)}
            className="flex items-center gap-2 bg-[#059669] hover:bg-[#047857] text-white font-semibold py-2 px-4 rounded-xl transition-all text-sm"
          >
            <Layers className="w-4 h-4" /> Vérifier tous séparément
          </button>
        )}
        <button
          onClick={onSendAnyway}
          className="text-sm font-medium text-amber-700 hover:text-amber-900 underline underline-offset-2"
        >
          Envoyer tel quel quand même
        </button>
      </div>
    </div>
  );
}
