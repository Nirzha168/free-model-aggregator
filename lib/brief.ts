// brief.ts — the trial brief, the provider catalog probe, and candidate ranking.
//
// The brief (MAKEFUN_PROMPT_FILE) is prose wrapped around three JSON documents,
// found by their `schema` member:
//   providers  free-model-aggregator-01/providers/v1 — the ten providers: id, wire,
//              auth scheme, key env var, the key, and the base path under the env.
//   ratings    free-model-aggregator-01/ratings/v1 — every free model's rating in
//              milli-points, context length, tools/vision/json capability, free flag.
//   workload   free-model-aggregator-01/workload/v1 — the 20 requests in arrival
//              order, each with api, stream flag, weight and requirements.
// The catalog is built by probing each provider's own catalog route in its own
// wire. The probe is the only thing that ever learns which models a provider
// lists and whether the listing calls them free; the ratings table is the
// sealed truth for rating and capabilities where it has a row for the model.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

export interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (e.g. /p/groq) or an absolute URL; joined to the env URL to dial. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface ManifestRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  providers: Provider[];
  ratings: RatingRow[];
  manifest: ManifestRequest[];
  error: string | null;
}

export interface ListedModel {
  id: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: Provider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/** One (provider, model) pair that may serve a request, ranked by rating. */
export interface Candidate {
  provider: Provider;
  model: ListedModel;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

// --- brief parsing -----------------------------------------------------------

/** Index of the `}` closing the object opening at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function parseProvider(v: unknown): Provider | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base =
    strOf(v.path) !== ''
      ? strOf(v.path)
      : strOf(v.base_path) !== ''
        ? strOf(v.base_path)
        : strOf(v.base_url) !== ''
          ? strOf(v.base_url)
          : strOf(v.base);
  return {
    id,
    name: strOf(v.name) !== '' ? strOf(v.name) : id,
    wire,
    auth,
    base,
    keyEnv: strOf(v.key_env),
    key: strOf(v.key),
  };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = strOf(v.model);
  const provider = strOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli:
      typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseManifestRow(v: unknown): ManifestRequest | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const req: Requirement = {
    tools: requires.tools === true,
    vision: requires.vision === true,
    json: requires.json === true,
  };
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) {
    req.minContext = requires.min_context;
  }
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: req,
  };
}

export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = parseRating(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = parseManifestRow(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, manifest, error: null };
}

// --- catalog probe -----------------------------------------------------------

function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id =
    wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  // A listing that prices everything at 0 is a free listing; one that says
  // nothing about price leaves the question to the ratings table.
  let free: boolean | null = boolOrNull(v.free);
  if (free === null && isRecord(v.pricing)) {
    const values = Object.values(v.pricing).flatMap((x) => {
      if (typeof x === 'number') return [x];
      if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
      return [];
    });
    if (values.length > 0) free = values.every((n) => Number.isFinite(n) && n === 0);
  }
  return {
    id,
    free,
    contextLength:
      numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

export async function probeProvider(envUrl: string, provider: Provider, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

export function ratingOf(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Free by the ratings table where it has a row, else by the listing, else paid. */
export function isFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

export interface ModelCaps {
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** Capabilities: the ratings row where it has one, the listing where it does not. */
export function capsOf(model: ListedModel, row: RatingRow | null): ModelCaps {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? false,
    vision: row?.vision ?? model.vision ?? false,
    json: row?.json ?? model.json ?? false,
  };
}

export function meets(requires: Requirement, caps: ModelCaps): boolean {
  if (requires.tools === true && !caps.tools) return false;
  if (requires.vision === true && !caps.vision) return false;
  if (requires.json === true && !caps.json) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) {
    return false;
  }
  return true;
}

/**
 * The catalog: every provider that answered, every model it listed, models
 * sorted by id. base_url is the provider's own path, never joined to the env
 * origin: the origin differs between a live pass and its replay, and the
 * catalog is part of the state export a replay must reproduce byte for byte.
 */
export function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOf(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFree(model, row),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/**
 * Every free (provider, model) pair that can serve `requires`, best rating
 * first, ties broken by provider id then model id — a fixed order, so a
 * replay makes the same picks. A JSON-mode request never goes to an
 * Anthropic-wire provider: that wire carries no response_format, so the
 * provider would echo json_mode absent and the request would fail its wire
 * checks however good the model is.
 */
export function candidatesFor(probes: readonly Probe[], ratings: RatingRow[], requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOf(ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      const caps = capsOf(model, row);
      if (!meets(requires, caps)) continue;
      out.push({
        provider: probe.provider,
        model,
        ratingMilli: row?.ratingMilli ?? 0,
        contextLength: caps.contextLength,
        tools: caps.tools,
        vision: caps.vision,
        json: caps.json,
      });
    }
  }
  out.sort((a, b) => {
    if (b.ratingMilli !== a.ratingMilli) return b.ratingMilli - a.ratingMilli;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}
