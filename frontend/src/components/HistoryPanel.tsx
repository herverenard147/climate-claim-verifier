import React, { useEffect, useState } from 'react';
import { History, Share2, X, Loader2, AlertCircle } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { shareOrCopy } from '../shareText';

interface HistoryEntry {
  id: number;
  claim: string;
  comprehension_level: string;
  badge_class: string;
  badge_icon: string;
  badge_text: string;
  analyse_text: string;
  sources: any[];
  created_at: string;
}

// Panneau de consultation de l'historique personnel (phase 3.2) : ne recalcule
// jamais rien, affiche les entrées telles qu'enregistrées par check-claim au
// moment de chaque vérification (voir main.py, history_store.py).
export default function HistoryPanel({ userId, onClose }: { userId: string; onClose: () => void }) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/history/${encodeURIComponent(userId)}`)
      .then((res) => {
        if (!res.ok) throw new Error("Impossible de charger l'historique.");
        return res.json();
      })
      .then(setEntries)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [userId]);

  const handleShareEntry = async (entry: HistoryEntry) => {
    await shareOrCopy(entry);
  };

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto p-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-bold text-[#1E293B] flex items-center gap-2">
            <History className="w-5 h-5" /> Mon historique
          </h2>
          <button onClick={onClose} className="text-[#94A3B8] hover:text-[#334155]">
            <X className="w-5 h-5" />
          </button>
        </div>

        {loading && (
          <p className="text-[#64748B] flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" /> Chargement de votre historique...
          </p>
        )}
        {error && (
          <p className="text-red-600 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" /> {error}
          </p>
        )}
        {!loading && !error && entries.length === 0 && (
          <p className="text-[#64748B]">Aucune vérification enregistrée pour l'instant.</p>
        )}

        <div className="space-y-4">
          {entries.map((entry) => (
            <div key={entry.id} className="border border-[#E2E8F0] rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#64748B] uppercase">
                  {entry.badge_icon} {entry.badge_text}
                </span>
                <span className="text-xs text-[#94A3B8]">
                  {new Date(entry.created_at).toLocaleString('fr-FR')} · niveau {entry.comprehension_level}
                </span>
              </div>
              <p className="text-[#334155] font-medium mb-2">« {entry.claim} »</p>
              <p className="text-sm text-[#64748B] mb-3">{entry.analyse_text}</p>
              <button
                onClick={() => handleShareEntry(entry)}
                className="flex items-center gap-1.5 text-xs font-semibold text-[#059669] hover:underline"
              >
                <Share2 className="w-3.5 h-3.5" /> Partager cette vérification
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
