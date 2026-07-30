import React, { useState } from 'react';
import { History, Layers } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ClaimInput from './components/ClaimInput';
import VerdictCard from './components/VerdictCard';
import SourcesAccordion from './components/SourcesAccordion';
import HistoryPanel from './components/HistoryPanel';
import BatchPanel from './components/BatchPanel';
import { API_BASE_URL } from './config';
import { getUserId } from './userId';
import { extractApiError } from './apiError';

// Un aller-retour API réussi peut être très rapide (quelques centaines de ms
// une fois les modèles chauds) : sans plancher, le spinner apparaît et
// disparaît trop vite pour être perçu, donnant l'impression qu'il ne s'est
// rien passé. On garantit qu'il reste visible au moins ce délai.
const MIN_LOADING_MS = 400;

export default function App() {
  const [claim, setClaim] = useState("");
  const [zoneGeo, setZoneGeo] = useState("Global (International)");
  const [comprehensionLevel, setComprehensionLevel] = useState("intermédiaire");
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const [showBatch, setShowBatch] = useState(false);
  const [userId] = useState(getUserId);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");

  const handleVerify = async (text: string) => {
    if (!text.trim()) return;
    setIsVerifying(true);
    setError("");
    setResult(null);
    const startedAt = Date.now();

    try {
      const response = await fetch(`${API_BASE_URL}/api/check-claim`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: text, zone_geo: zoneGeo, comprehension_level: comprehensionLevel, user_id: userId })
      });
      const data = await response.json();
      // Le corps de la réponse est lu AVANT de vérifier response.ok : sinon
      // le message d'erreur précis renvoyé par le backend (ex. "La
      // déclaration est vide.") est perdu et remplacé par un message
      // générique, alors qu'il était déjà disponible - même défaut que celui
      // corrigé sur l'upload de document.
      if (!response.ok) throw new Error(extractApiError(data, "Erreur serveur lors de la vérification."));
      // On rattache la déclaration vérifiée à `result` (l'API ne la renvoie pas
      // elle-même) : nécessaire pour que l'export/partage du verdict puisse
      // citer l'affirmation d'origine sans dépendre du state `claim`, qui peut
      // changer entre-temps si l'utilisateur retape quelque chose.
      setResult({ ...data, claim: text });
    } catch (err: any) {
      setError(err.message);
    } finally {
      const elapsed = Date.now() - startedAt;
      if (elapsed < MIN_LOADING_MS) {
        await new Promise((resolve) => setTimeout(resolve, MIN_LOADING_MS - elapsed));
      }
      setIsVerifying(false);
    }
  };

  const handlePdfUpload = async (file: File) => {
    // Un vrai PDF (rapport GIEC, scan) peut prendre plusieurs secondes à
    // traiter côté serveur (extraction + OCR de secours page par page) :
    // sans état de chargement explicite, l'utilisateur ne voit strictement
    // rien se passer pendant tout ce temps (constaté : 8,4s de silence total
    // sur un PDF réel de 7 Mo avant ce correctif).
    setIsUploading(true);
    setUploadError("");
    setUploadSuccess("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload-pdf`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) throw new Error(extractApiError(data, "Erreur lors de la lecture du document."));
      setClaim(data.extracted_text);
      // Le champ ne reçoit qu'un aperçu (début + fin) quand le document est
      // long : c'est aussi ce qui sera envoyé à la vérification, pas le
      // document entier - le préciser pour ne pas laisser croire que
      // l'intégralité du texte a été prise en compte.
      setUploadSuccess(
        data.truncated
          ? `« ${file.name} » est plus long que l'aperçu inséré : seuls le début et la fin du texte extrait ont été copiés ci-dessous. Complétez-le si besoin avant de vérifier.`
          : `« ${file.name} » analysé : le texte extrait a été inséré ci-dessous.`
      );
    } catch (err: any) {
      // Une erreur réseau (backend injoignable, mauvais port, CORS...) ne
      // produit pas de message exploitable côté navigateur (juste "Failed to
      // fetch") : on l'affiche quand même mais avec un message plus clair
      // que le message brut, pour que l'utilisateur comprenne qu'il s'agit
      // d'un problème de connexion au serveur et non du document lui-même.
      const isNetworkError = err instanceof TypeError;
      setUploadError(
        isNetworkError
          ? "Impossible de contacter le serveur d'analyse. Vérifiez que le backend est bien démarré et accessible."
          : err.message
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#F8FAFC] text-[#0F172A] font-inter">
      <Sidebar
        zoneGeo={zoneGeo}
        setZoneGeo={setZoneGeo}
        comprehensionLevel={comprehensionLevel}
        setComprehensionLevel={setComprehensionLevel}
        onPdfUpload={handlePdfUpload}
        isUploading={isUploading}
        uploadError={uploadError}
        uploadSuccess={uploadSuccess}
      />
      
      <main className="flex-1 overflow-y-auto p-8 lg:p-12">
        <div className="max-w-4xl mx-auto space-y-8">
          
          <header className="mb-12 flex items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-extrabold tracking-tight mb-2">Analyse Climatique</h1>
              <p className="text-[#64748B] text-lg">Vérifiez la validité scientifique d'une déclaration par rapport aux données du GIEC et de l'OMM.</p>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <button
                onClick={() => setShowBatch(true)}
                className="flex items-center gap-2 bg-white border border-[#E2E8F0] hover:bg-[#F8FAFC] text-[#334155] font-semibold py-2.5 px-4 rounded-xl shadow-sm transition-all"
              >
                <Layers className="w-4 h-4" /> Vérification par lot
              </button>
              <button
                onClick={() => setShowHistory(true)}
                className="flex items-center gap-2 bg-white border border-[#E2E8F0] hover:bg-[#F8FAFC] text-[#334155] font-semibold py-2.5 px-4 rounded-xl shadow-sm transition-all"
              >
                <History className="w-4 h-4" /> Mon historique
              </button>
            </div>
          </header>

          {showHistory && <HistoryPanel userId={userId} onClose={() => setShowHistory(false)} />}
          {showBatch && (
            <BatchPanel
              zoneGeo={zoneGeo}
              comprehensionLevel={comprehensionLevel}
              userId={userId}
              onClose={() => setShowBatch(false)}
            />
          )}

          <ClaimInput claim={claim} setClaim={setClaim} onVerify={handleVerify} isLoading={isVerifying} />
          
          {error && (
            <div className="bg-red-50 text-red-700 p-4 rounded-xl border border-red-200">
              {error}
            </div>
          )}

          {result && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <VerdictCard result={result} userId={userId} />
              <SourcesAccordion sources={result.sources} />
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
