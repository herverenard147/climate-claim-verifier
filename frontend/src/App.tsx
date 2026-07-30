import React, { useState } from 'react';
import { History, Layers } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ClaimInput from './components/ClaimInput';
import VerdictCard from './components/VerdictCard';
import SourcesAccordion from './components/SourcesAccordion';
import HistoryPanel from './components/HistoryPanel';
import BatchPanel from './components/BatchPanel';
import GuidanceCard from './components/GuidanceCard';
import { API_BASE_URL } from './config';
import { getUserId } from './userId';
import { extractApiError } from './apiError';

// Un aller-retour API réussi peut être très rapide (quelques centaines de ms
// une fois les modèles chauds) : sans plancher, le spinner apparaît et
// disparaît trop vite pour être perçu, donnant l'impression qu'il ne s'est
// rien passé. On garantit qu'il reste visible au moins ce délai.
const MIN_LOADING_MS = 400;

// Doivent rester synchronisées avec MAX_UPLOAD_SIZE_BYTES / MAX_OCR_PAGES
// dans main.py. Valeurs déterminées par des tests réels contre Render (pas
// choisies par prudence) : 6,6-6,9 Mo fonctionne de façon répétée, 9,85 Mo
// fait planter le service à coup sûr (502, process redémarré) - 8 Mo laisse
// une marge des deux côtés. MAX_OCR_PAGES=0 : même 1 seule page OCR sur un
// fichier de 23 Ko a fait planter le service en ~5s - l'OCR n'est pas
// seulement plafonné, il est désactivé en production sur ce plan Render
// (voir DOCUMENTATION_TECHNIQUE.md pour la méthodologie complète). Le
// contrôle de taille ici évite un aller-retour réseau inutile pour un
// fichier déjà trop gros ; le backend revalide de toute façon (ce contrôle
// client ne suffit pas seul, un utilisateur pourrait le contourner).
const MAX_UPLOAD_SIZE_BYTES = 8 * 1024 * 1024;
const MAX_OCR_PAGES = 0;

export default function App() {
  const [claim, setClaim] = useState("");
  const [zoneGeo, setZoneGeo] = useState("Global (International)");
  const [comprehensionLevel, setComprehensionLevel] = useState("intermédiaire");
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [showHistory, setShowHistory] = useState(false);
  const [showBatch, setShowBatch] = useState(false);
  const [batchInitialText, setBatchInitialText] = useState<string | undefined>(undefined);
  const [guidance, setGuidance] = useState<any>(null);
  const [userId] = useState(getUserId);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");
  const [uploadPartial, setUploadPartial] = useState(false);

  const handleVerify = async (text: string, force: boolean = false) => {
    if (!text.trim()) return;
    setIsVerifying(true);
    setError("");
    setResult(null);
    setGuidance(null);
    const startedAt = Date.now();

    try {
      const response = await fetch(`${API_BASE_URL}/api/check-claim`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: text, zone_geo: zoneGeo, comprehension_level: comprehensionLevel, user_id: userId, force })
      });
      const data = await response.json();
      // Le corps de la réponse est lu AVANT de vérifier response.ok : sinon
      // le message d'erreur précis renvoyé par le backend (ex. "La
      // déclaration est vide.") est perdu et remplacé par un message
      // générique, alors qu'il était déjà disponible - même défaut que celui
      // corrigé sur l'upload de document.
      if (!response.ok) throw new Error(extractApiError(data, "Erreur serveur lors de la vérification."));

      // needs_guidance : détection heuristique (saisie multiple/vague/
      // incohérente, voir input_heuristics.py côté backend) - pas un verdict,
      // on affiche des choix fermés plutôt qu'un résultat trompeur. Le texte
      // d'origine est conservé pour permettre "envoyer tel quel quand même".
      if (data.needs_guidance) {
        setGuidance({ ...data, originalText: text });
        return;
      }

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

  const handleVerifyAllSeparately = (segments: string[]) => {
    setGuidance(null);
    setBatchInitialText(segments.join("\n"));
    setShowBatch(true);
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
    setUploadPartial(false);

    // Rejet immédiat côté client, sans aller-retour réseau, pour un fichier
    // déjà trop gros - le backend revalide de toute façon (voir plus haut).
    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      setUploadError(
        `Fichier trop volumineux (${(file.size / (1024 * 1024)).toFixed(1)} Mo, max ${MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)} Mo).`
      );
      setIsUploading(false);
      return;
    }

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
      const messages: string[] = [
        data.truncated
          ? `« ${file.name} » est plus long que l'aperçu inséré : seuls le début et la fin du texte extrait ont été copiés ci-dessous.`
          : `« ${file.name} » analysé : le texte extrait a été inséré ci-dessous.`,
      ];

      // pages_failed n'existe que pour un PDF (null pour un .txt, voir
      // main.py) : signale honnêtement les pages non exploitées plutôt que
      // de laisser croire que tout le document a été lu avec succès.
      const hasPageIssue = typeof data.pages_failed === "number" && data.pages_failed > 0;
      if (hasPageIssue) {
        messages.push(
          data.ocr_capped
            ? (MAX_OCR_PAGES > 0
                ? `${data.pages_failed} page(s) sur ${data.pages_total} n'ont pas pu être lues (limite de reconnaissance de texte scanné : ${MAX_OCR_PAGES} pages max par document).`
                : `${data.pages_failed} page(s) sur ${data.pages_total} n'ont pas pu être lues : la reconnaissance de texte scanné (OCR) n'est pas disponible sur cet environnement.`)
            : `${data.pages_failed} page(s) sur ${data.pages_total} n'ont pas pu être lues (page vide ou scan illisible).`
        );
      }
      setUploadPartial(hasPageIssue);
      setUploadSuccess(messages.join(" "));
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
        uploadPartial={uploadPartial}
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
                onClick={() => { setBatchInitialText(undefined); setShowBatch(true); }}
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
              initialText={batchInitialText}
            />
          )}

          <ClaimInput claim={claim} setClaim={setClaim} onVerify={handleVerify} isLoading={isVerifying} />

          {guidance && (
            <GuidanceCard
              guidance={guidance}
              onVerifySegment={(segment) => handleVerify(segment)}
              onVerifyAllSeparately={handleVerifyAllSeparately}
              onSendAnyway={() => handleVerify(guidance.originalText, true)}
            />
          )}

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
