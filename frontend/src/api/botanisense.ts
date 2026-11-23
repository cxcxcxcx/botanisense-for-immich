import type { User } from 'firebase/auth';
import { BOTANISENSE_API_URL, type BotanisenseDetection } from '../lib/botanisense';

export class BotanisenseApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
    this.name = 'BotanisenseApiError';
  }
}

export interface DeviceCodeStart {
  device_code: string;
  user_code: string;
  verification_uri: string;
  expires_in: number;
  interval: number;
}

export interface DeviceTokenApproved {
  status: 'approved';
  customToken: string;
  uid: string;
  email: string | null;
  displayName: string | null;
  photoUrl: string | null;
  instanceName: string | null;
  scopes: string[];
}

export type DeviceTokenPoll =
  | { status: 'pending' }
  | DeviceTokenApproved;

export interface BotanisenseImportResult {
  photoId: string;
  imagePath: string;
  url: string;
  deduped: boolean;
  entryId: string | null;
  detectedSpeciesIds: string[];
}

export interface BotanisenseMe {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoUrl: string | null;
  grant: {
    provider: string | null;
    instanceName: string | null;
    instanceId: string | null;
    approvedAt: string | null;
    revoked: boolean;
    importedPhotos: number;
    lastImportAt: string | null;
  } | null;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === 'object' && 'error' in body) {
    const value = (body as { error?: unknown }).error;
    if (typeof value === 'string' && value) return value;
  }
  return fallback;
}

export function botanisenseInstanceId(): string {
  const key = 'botanisense:instance-id';
  try {
    const existing = localStorage.getItem(key);
    if (existing) return existing;
    const generated =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem(key, generated);
    return generated;
  } catch {
    return `ephemeral-${Math.random().toString(36).slice(2, 10)}`;
  }
}

export function botanisenseInstanceName(): string {
  if (typeof window === 'undefined') return 'BotaniSense for Immich';
  return `BotaniSense for Immich (${window.location.host})`;
}

export async function startBotanisenseDeviceCode(): Promise<DeviceCodeStart> {
  const response = await fetch(`${BOTANISENSE_API_URL}/api/integrations/device/code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      instanceName: botanisenseInstanceName(),
      instanceId: botanisenseInstanceId(),
    }),
  });
  const body = await readJson(response);
  if (!response.ok) {
    throw new BotanisenseApiError(
      response.status,
      errorMessage(body, `Could not start BotaniSense login (HTTP ${response.status})`),
      body,
    );
  }
  return body as DeviceCodeStart;
}

export async function pollBotanisenseDeviceToken(deviceCode: string): Promise<DeviceTokenPoll> {
  const response = await fetch(`${BOTANISENSE_API_URL}/api/integrations/device/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ deviceCode }),
  });
  const body = await readJson(response);
  if (response.status === 202) return { status: 'pending' };
  if (!response.ok) {
    throw new BotanisenseApiError(
      response.status,
      errorMessage(body, `Could not finish BotaniSense login (HTTP ${response.status})`),
      body,
    );
  }
  return { ...(body as Omit<DeviceTokenApproved, 'status'>), status: 'approved' };
}

async function withUserToken<T>(user: User, call: (token: string) => Promise<T>): Promise<T> {
  const token = await user.getIdToken();
  try {
    return await call(token);
  } catch (error) {
    if (error instanceof BotanisenseApiError && error.status === 401) {
      return call(await user.getIdToken(true));
    }
    throw error;
  }
}

export async function fetchBotanisenseMe(user: User): Promise<BotanisenseMe> {
  return withUserToken(user, async token => {
    const response = await fetch(`${BOTANISENSE_API_URL}/api/integrations/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    const body = await readJson(response);
    if (!response.ok) {
      throw new BotanisenseApiError(
        response.status,
        errorMessage(body, `Could not read BotaniSense account (HTTP ${response.status})`),
        body,
      );
    }
    return body as BotanisenseMe;
  });
}

export async function revokeBotanisenseIntegration(
  user: User,
  revokeSessions = false,
): Promise<void> {
  await withUserToken(user, async token => {
    const response = await fetch(`${BOTANISENSE_API_URL}/api/integrations/revoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ revokeSessions }),
    });
    if (!response.ok) {
      const body = await readJson(response);
      throw new BotanisenseApiError(
        response.status,
        errorMessage(body, `Could not disconnect BotaniSense (HTTP ${response.status})`),
        body,
      );
    }
  });
}

export interface ImportPhotoParams {
  file: Blob;
  filename: string;
  clientRef: string;
  capturedAt?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  detections: BotanisenseDetection[];
}

/**
 * Upload one photo + the local BioCLIP results. The server writes the photo
 * document and species connections; it never calls Gemini when detections are
 * present.
 */
export async function importPhotoToBotanisense(
  user: User,
  params: ImportPhotoParams,
): Promise<BotanisenseImportResult> {
  const form = new FormData();
  form.append('file', params.file, params.filename);
  form.append('clientRef', params.clientRef);
  if (params.capturedAt) form.append('capturedAt', params.capturedAt);
  if (params.latitude != null) form.append('latitude', String(params.latitude));
  if (params.longitude != null) form.append('longitude', String(params.longitude));
  form.append('detections', JSON.stringify(params.detections));

  return withUserToken(user, async token => {
    const response = await fetch(`${BOTANISENSE_API_URL}/api/integrations/import`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
    const body = await readJson(response);
    if (!response.ok) {
      throw new BotanisenseApiError(
        response.status,
        errorMessage(body, `Import failed (HTTP ${response.status})`),
        body,
      );
    }
    return body as BotanisenseImportResult;
  });
}
