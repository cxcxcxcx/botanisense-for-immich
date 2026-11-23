// BotaniSense (official Firebase app) client bindings.
//
// The user signs in as *their own* account: the BotaniSense backend mints a
// Firebase custom token for their uid after they approve the connection on
// botanisense.app. `signInWithCustomToken` then gives us a normal Firebase
// session, so reads go straight to Firestore under the existing owner rules.
//
// This file only contains public configuration. The Firebase web API key is
// not a secret; it is the same key the official app ships.
import { getApps, initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import type { AssetDetail, PlantAttributionRow } from '../api/types';

const firebaseConfig = {
  apiKey:
    import.meta.env.VITE_BOTANISENSE_FIREBASE_API_KEY ??
    'AIzaSyCMb0QvG9Jx9Hc2yAPqYPLXgPN6ojzXs-g',
  authDomain:
    import.meta.env.VITE_BOTANISENSE_AUTH_DOMAIN ??
    'plantsun-7b88e.firebaseapp.com',
  projectId:
    import.meta.env.VITE_BOTANISENSE_PROJECT_ID ?? 'plantsun-7b88e',
  storageBucket:
    import.meta.env.VITE_BOTANISENSE_STORAGE_BUCKET ??
    'plantsun-7b88e.firebasestorage.app',
  messagingSenderId:
    import.meta.env.VITE_BOTANISENSE_MESSAGING_SENDER_ID ?? '394537748734',
  appId:
    import.meta.env.VITE_BOTANISENSE_APP_ID ??
    '1:394537748734:web:d855a2f037f6d30f28bdaf',
};

export const BOTANISENSE_API_URL =
  import.meta.env.VITE_BOTANISENSE_API_URL ?? 'https://api.botanisense.app';

export const BOTANISENSE_IMAGEKIT_URL =
  import.meta.env.VITE_BOTANISENSE_IMAGEKIT_URL ??
  'https://ik.imagekit.io/bpgiejhtz/';

export const BOTANISENSE_WEB_URL =
  import.meta.env.VITE_BOTANISENSE_WEB_URL ?? 'https://botanisense.app';

export const botanisenseApp =
  getApps().find(app => app.name === 'botanisense') ??
  initializeApp(firebaseConfig, 'botanisense');
export const botanisenseAuth = getAuth(botanisenseApp);
export const botanisenseDb = getFirestore(botanisenseApp);

/** ImageKit serves the paths stored on photo docs; full URLs pass through. */
export function botanisenseImageUrl(pathOrUrl: string | null | undefined): string {
  const value = (pathOrUrl ?? '').trim();
  if (!value) return '';
  if (/^https?:\/\//i.test(value) || value.startsWith('blob:') || value.startsWith('data:')) {
    return value;
  }
  return BOTANISENSE_IMAGEKIT_URL + value.replace(/^\/+/, '');
}

export interface BotanisenseDetection {
  LatinName: string;
  Confidence: number;
  commonNames?: string[];
  '中文名称'?: string;
  pot: 'in-ground' | 'potted' | 'unknown';
  location: 'plant shop' | 'indoor' | 'garden' | 'roadside' | 'wild' | 'unknown';
  context: 'close-up' | 'landscape' | 'room' | 'unknown';
  plant_visibility: string[];
  ConfidenceRationale: string;
  box_2d: number[];
  source: string;
}

function rowToDetection(
  row: PlantAttributionRow,
  detail: AssetDetail,
): BotanisenseDetection | null {
  const latin = (row.species ?? '').trim();
  if (!latin || latin.toLowerCase() === 'unknown') return null;

  let box = [0, 0, 1000, 1000];
  if (row.detection_id != null) {
    const detection = detail.detections?.find(d => d.id === row.detection_id);
    if (detection) {
      box = [
        Math.round(detection.y * 1000),
        Math.round(detection.x * 1000),
        Math.round((detection.y + detection.h) * 1000),
        Math.round((detection.x + detection.w) * 1000),
      ];
    }
  }

  return {
    LatinName: latin,
    Confidence: Number.isFinite(row.score) ? row.score : 0,
    ...(row.common_name ? { commonNames: [row.common_name] } : {}),
    '中文名称': '',
    pot: 'unknown',
    location: 'unknown',
    context: 'unknown',
    plant_visibility: ['whole-plant'],
    ConfidenceRationale: 'BioCLIP-2 zero-shot (Immich bridge)',
    box_2d: box,
    source: 'bioclip-2',
  };
}

/**
 * Map this app's attribution rows to BotaniSense's `PlantIdentification[]`.
 *
 * One entry = one plant in the photo, not the whole top-K list. If YOLO boxes
 * exist we use their top-1 per box; otherwise the whole-image top-1. Writing
 * all five BioCLIP candidates would tell BotaniSense there are five plants
 * and inflate every species count.
 */
export function toBotanisenseDetections(detail: AssetDetail): BotanisenseDetection[] {
  const rows = detail.plant_attribution?.rows ?? [];
  const boxed = rows.filter(row => row.detection_id != null);
  const candidates = boxed.length > 0 ? boxed : rows.slice(0, 1);

  const detections: BotanisenseDetection[] = [];
  const seen = new Set<string>();
  for (const row of candidates) {
    const detection = rowToDetection(row, detail);
    if (!detection) continue;
    const key = `${detection.LatinName}|${detection.box_2d.join(',')}`;
    if (seen.has(key)) continue;
    seen.add(key);
    detections.push(detection);
  }

  if (detections.length === 0) {
    const top = detail.top_predictions?.[0];
    if (top && top.species && top.species.toLowerCase() !== 'unknown') {
      detections.push({
        LatinName: top.species,
        Confidence: Number.isFinite(top.score) ? top.score : 0,
        ...(top.common_name ? { commonNames: [top.common_name] } : {}),
        '中文名称': '',
        pot: 'unknown',
        location: 'unknown',
        context: 'unknown',
        plant_visibility: ['whole-plant'],
        ConfidenceRationale: 'BioCLIP-2 zero-shot (Immich bridge)',
        box_2d: [0, 0, 1000, 1000],
        source: 'bioclip-2',
      });
    }
  }

  return detections;
}
