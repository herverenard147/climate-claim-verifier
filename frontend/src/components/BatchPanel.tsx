import React, { useState } from 'react';
import { Layers, X, Loader2, AlertCircle } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { extractApiError } from '../apiError';

interface BatchResultItem {
  claim: string;
  result: any;
  error?: string;
}

// Limite défensive côté client, alignée sur MAX_BATCH_CLAIMS dans main.py
// (l'appel se fait désormais ligne par ligne contre /api/check-claim, voir
// note ci-dessous, donc ce plafond n'est plus imposé par le backend lui-même
// pour cette route mais reste utile pour éviter un lot de saisie absurde).
const MAX_BATCH_CLAIMS = 20;

// Vérification par lot (phase 5) : un texte collé, une affirmation par
// ligne (découpage naïf, pas de segmentation NLP par phrase - voir
// check_claims_batch dans main.py pour les limites assumées de ce
// découpage).
//
// Appelle /api/check-claim ligne par ligne (au lieu de l'endpoint dédié
// /api/check-claims-batch qui traite tout le lot en une seule réponse) pour
// pouvoir afficher une progression réelle "x/N traités" pendant l'attente,
// plutôt qu'un silence total jusqu'au résultat complet — les deux chemins
// exécutent EXACTEMENT le même pipeline par claim côté backend
// (check_claims_batch appelle lui-même check_claim() ligne par ligne).
export default function BatchPanel({
  zoneGeo,
  comprehensionLevel,
  userId,
  onClose,
  initialText,
}: {
  zoneGeo: string;
  comprehensionLevel: string;
  userId: string;
  onClose: () => void;
  // Pré-remplissage optionnel (ex. depuis le guidage "saisie multiple" du
  // champ principal : l'utilisateur choisit "vérifier tous séparément" et
  // retrouve directement ses affirmations détectées, une par ligne).
  initialText?: string;
}) {
  const [text, setText] = useState(initialText ?? '');
  const [results, setResults] = useState<BatchResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState({ done: 0, total: 0 });

  const handleSubmit = async () => {
    const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
    if (lines.length === 0) return;
    if (lines.length > MAX_BATCH_CLAIMS) {
      setError(`Trop d'affirmations dans le lot (${lines.length}) : maximum ${MAX_BATCH_CLAIMS} par vérification.`);
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);
    setProgress({ done: 0, total: lines.length });

    const collected: BatchResultItem[] = [];
    for (const line of lines) {
      try {
        const res = await fetch(`${API_BASE_URL}/api/check-claim`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          // force: true - une ligne du lot est déjà une affirmation séparée
          // par construction ("une ligne = un claim") : sans ce flag, la
          // détection de saisie multiple/vague (voir main.py,
          // input_heuristics.py) renverrait needs_guidance au lieu d'un
          // verdict pour toute ligne courte ou tournée en question, que ce
          // composant ne sait pas afficher (il attend directement un
          // verdict par ligne, pas de guidage interactif).
          body: JSON.stringify({ claim: line, zone_geo: zoneGeo, comprehension_level: comprehensionLevel, user_id: userId, force: true }),
        });
        const data = await res.json();
        if (!res.ok) {
          collected.push({ claim: line, result: null, error: extractApiError(data, 'Erreur lors de la vérification de cette affirmation.') });
        } else {
          collected.push({ claim: line, result: data });
        }
      } catch (err: any) {
        collected.push({ claim: line, result: null, error: 'Impossible de contacter le serveur pour cette affirmation.' });
      }
      setProgress((p) => ({ ...p, done: p.done + 1 }));
      setResults([...collected]);
    }

    setLoading(false);
  };

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-2xl w-full max-h-[85vh] overflow-y-auto p-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-[#1E293B] flex items-center gap-2">
            <Layers className="w-5 h-5" /> Vérification par lot
          </h2>
          <button onClick={onClose} className="text-[#94A3B8] hover:text-[#334155]">
            <X className="w-5 h-5" />
          </button>
        </div>
        <p className="text-sm text-[#64748B] mb-1">
          Utile pour vérifier d'un coup plusieurs affirmations d'un article ou d'un fil de discussion,
          plutôt qu'une par une.
        </p>
        <p className="text-sm text-[#64748B] mb-4">
          Collez vos affirmations ci-dessous, <strong>une par ligne</strong> (max {MAX_BATCH_CLAIMS}).
          Chacune suit le pipeline complet de vérification, l'une après l'autre.
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={
            "Les températures moyennes en Côte d'Ivoire ont augmenté depuis 1960\n" +
            "Le niveau de la mer monte plus vite en Afrique de l'Ouest que la moyenne mondiale\n" +
            "La Terre est plate"
          }
          disabled={loading}
          className="w-full min-h-[140px] border border-[#CBD5E1] rounded-lg p-3 text-sm mb-1 focus:outline-none focus:ring-2 focus:ring-[#059669]/20 focus:border-[#059669] disabled:bg-[#F8FAFC] disabled:text-[#94A3B8]"
        />
        {/* Le placeholder disparaît dès la première frappe : ce rappel du
            format reste visible pendant toute la saisie, pas seulement avant. */}
        <p className="text-xs text-[#94A3B8] mb-4">Une affirmation par ligne — pas de question, pas de puces.</p>

        <button
          onClick={handleSubmit}
          disabled={loading || !text.trim()}
          className="flex items-center gap-2 bg-[#059669] hover:bg-[#047857] disabled:opacity-50 text-white font-semibold py-2.5 px-5 rounded-xl transition-all mb-2"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Layers className="w-4 h-4" />}
          {loading ? `Vérification en cours... (${progress.done}/${progress.total} traités)` : 'Vérifier le lot'}
        </button>

        {loading && (
          <div className="w-full bg-[#F1F5F9] rounded-full h-1.5 mb-6 overflow-hidden">
            <div
              className="bg-[#059669] h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }}
            />
          </div>
        )}
        {!loading && <div className="mb-6" />}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2 mb-4">
            <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {results && results.length > 0 && (
          <div className="space-y-3">
            {results.map((item, idx) =>
              item.error ? (
                <div key={idx} className="border border-red-200 bg-red-50 rounded-xl p-4">
                  <p className="text-xs font-bold text-red-700 uppercase mb-1 flex items-center gap-1.5">
                    <AlertCircle className="w-3.5 h-3.5" /> Échec de la vérification
                  </p>
                  <p className="text-sm text-[#334155] mb-1">« {item.claim} »</p>
                  <p className="text-xs text-red-700">{item.error}</p>
                </div>
              ) : (
                <div key={idx} className="border border-[#E2E8F0] rounded-xl p-4">
                  <p className="text-xs font-bold text-[#64748B] uppercase mb-1">
                    {item.result.badge_icon} {item.result.badge_text}
                  </p>
                  <p className="text-sm text-[#334155]">« {item.claim} »</p>
                </div>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
