// Le backend renvoie `detail` soit comme chaîne lisible (HTTPException levée
// explicitement dans le code métier, ex. "La déclaration est vide."), soit
// comme tableau d'objets (erreur de validation Pydantic générée
// automatiquement, ex. champ requis manquant) — jamais affichable tel quel
// dans ce second cas. Centralisé ici car plusieurs endroits du frontend
// appelaient `data.detail || fallback` directement et perdaient le message
// réel du backend (montré tel quel côté utilisateur ou silencieusement
// remplacé par un message générique moins utile).
export function extractApiError(data: any, fallback: string): string {
  const detail = data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length > 0 && typeof detail[0]?.msg === "string") {
    return detail[0].msg;
  }
  return fallback;
}
