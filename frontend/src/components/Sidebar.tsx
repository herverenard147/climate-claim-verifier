import React, { useRef, useState } from 'react';
import { Globe, UploadCloud, CheckCircle2, Loader2, AlertCircle } from 'lucide-react';

const LEVEL_DESCRIPTIONS: Record<string, string> = {
  "débutant": "Explication simple et vulgarisée, sans jargon.",
  "intermédiaire": "Explication claire avec contexte (niveau par défaut).",
  "amateur": "Ajoute les scores et le nombre de sources utilisés.",
  "expert": "Sortie technique brute : classification NLI, score cosinus, probabilités par classe.",
};

interface SidebarProps {
  zoneGeo: string;
  setZoneGeo: (zone: string) => void;
  comprehensionLevel: string;
  setComprehensionLevel: (level: string) => void;
  onPdfUpload: (file: File) => void;
  isUploading: boolean;
  uploadError: string;
  uploadSuccess: string;
}

export default function Sidebar({
  zoneGeo,
  setZoneGeo,
  comprehensionLevel,
  setComprehensionLevel,
  onPdfUpload,
  isUploading,
  uploadError,
  uploadSuccess,
}: SidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onPdfUpload(e.target.files[0]);
      e.target.value = ""; // permet de ré-uploader le même fichier à la suite (ex. après une erreur)
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (isUploading) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (isUploading) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onPdfUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <aside className="w-80 bg-white border-r border-[#E2E8F0] p-6 flex flex-col h-full shadow-sm">
      {/* Brand Header */}
      <div className="flex items-center gap-3 mb-12">
        <Globe className="text-[#059669] w-10 h-10" />
        <div>
          <h1 className="text-xl font-extrabold text-[#0F172A] leading-tight">TERRAVA-AI <span className="text-[#059669]">Pro</span></h1>
          <p className="text-xs text-[#64748B] font-medium">Fact-Checking Climatique</p>
        </div>
      </div>

      {/* Filtre Régional */}
      <div className="mb-10">
        <label className="block text-sm font-bold text-[#1E293B] uppercase tracking-wide mb-3">📍 Zone Géographique</label>
        <select 
          value={zoneGeo} 
          onChange={(e) => setZoneGeo(e.target.value)}
          className="w-full bg-[#F8FAFC] border border-[#CBD5E1] rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[#059669]/20 focus:border-[#059669] transition-all cursor-pointer"
        >
          <option>Global (International)</option>
          <option>Afrique de l'Ouest</option>
          <option>Côte d'Ivoire</option>
        </select>
      </div>

      {/* Niveau de compréhension */}
      <div className="mb-10">
        <label className="block text-sm font-bold text-[#1E293B] uppercase tracking-wide mb-3">🎓 Niveau de compréhension</label>
        <select
          value={comprehensionLevel}
          onChange={(e) => setComprehensionLevel(e.target.value)}
          className="w-full bg-[#F8FAFC] border border-[#CBD5E1] rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[#059669]/20 focus:border-[#059669] transition-all cursor-pointer"
        >
          <option value="débutant">Débutant</option>
          <option value="intermédiaire">Intermédiaire</option>
          <option value="amateur">Amateur</option>
          <option value="expert">Expert</option>
        </select>
        {/* Descriptions alignées sur ce que produit réellement chaque niveau
            côté backend (voir build_analyse_text dans main.py), pas des
            libellés inventés : débutant vulgarise, intermédiaire donne le
            contexte, amateur ajoute les scores, expert donne la sortie
            technique brute (classification NLI, probabilités). */}
        <p className="text-xs text-[#64748B] mt-2 leading-relaxed">
          {LEVEL_DESCRIPTIONS[comprehensionLevel]}
        </p>
      </div>

      {/* Analyse PDF */}
      <div>
        <label className="block text-sm font-bold text-[#1E293B] uppercase tracking-wide mb-3">📄 Analyse de Document</label>
        <p className="text-xs text-[#64748B] mb-4">
          Formats acceptés : PDF, TXT. Le texte extrait pré-remplit automatiquement le champ de vérification — vous pourrez le modifier avant de lancer la vérification.
        </p>
        
        <div
          onClick={() => !isUploading && fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center transition-all group ${
            isUploading
              ? "bg-[#F8FAFC] border-[#CBD5E1] cursor-wait"
              : isDragging
              ? "bg-[#F0FDF4] border-[#059669] cursor-pointer"
              : "border-[#CBD5E1] hover:bg-[#F8FAFC] hover:border-[#059669] cursor-pointer"
          }`}
        >
          {isUploading ? (
            <>
              <Loader2 className="w-8 h-8 text-[#059669] mb-2 animate-spin" />
              <span className="text-sm font-medium text-[#475569]">Analyse du document en cours...</span>
              <span className="text-xs text-[#94A3B8] mt-1">Cela peut prendre plusieurs secondes pour un PDF volumineux</span>
            </>
          ) : (
            <>
              <UploadCloud className="w-8 h-8 text-[#94A3B8] mb-2 group-hover:text-[#059669] transition-colors" />
              <span className="text-sm font-medium text-[#475569]">Glissez un PDF ou TXT ici ou</span>
              <span className="text-sm font-bold text-[#059669] mt-1">Parcourez vos fichiers</span>
            </>
          )}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,.txt"
            disabled={isUploading}
            className="hidden"
          />
        </div>

        {uploadError && (
          <div className="mt-3 bg-red-50 border border-red-200 rounded-lg p-3 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-red-700 font-medium leading-relaxed">{uploadError}</p>
          </div>
        )}

        {uploadSuccess && !isUploading && (
          <div className="mt-3 bg-[#F0FDF4] border border-[#BBF7D0] rounded-lg p-3 flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-[#16A34A] flex-shrink-0 mt-0.5" />
            <p className="text-xs text-[#166534] font-medium leading-relaxed">{uploadSuccess}</p>
          </div>
        )}
      </div>
      
      <div className="mt-auto">
        <div className="bg-[#F0FDF4] border border-[#BBF7D0] rounded-lg p-4 flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-[#16A34A] flex-shrink-0 mt-0.5" />
          <p className="text-xs text-[#166534] font-medium leading-relaxed">
            Connecté au moteur d'intelligence climatique. Données synchronisées.
          </p>
        </div>
      </div>
    </aside>
  );
}
