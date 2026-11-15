// main.ts — free-model-aggregator: one server, two client wires, ten providers.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic),
// JSON and SSE, plus GET /v1/models in the shape the auth header names.
// For each request it picks the best free model that meets the request's
// requirements, speaks that provider's wire, and on a 402/429/5xx refusal
// answers the request with the next-best provider instead. The pick is a pure
// function of the brief and the providers' answers so far: no wall clock, no
// unseeded randomness, so a replay leaves the same state export.
//
// Routing, in one paragraph: candidates are ranked by the sealed rating,
// best first. A request tries each provider at most once, in that order — a
// refusal is per provider (its budget, its failure schedule, its outage), so
// a second model on the same refused provider would refuse again and spend
// more of its budget. The next request starts from the top again: a single
// refused call may be the provider's scheduled failure, not a spent budget,
// and retrying costs only a tie-break call while skipping a live top model
// would cost quality on every request after it. A provider that refuses
// three attempts in a row is treated as spent for the rest of the trial
// (the generator schedules at most a pair of consecutive failures, so three
// in a row means budget spent or outage begun); that keeps the call count
// — the tie-break — down without giving up any live top model.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  candidatesFor,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type Brief,
  type Candidate,
  type Catalog,
  type Json,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  convertJson,
  convertStream,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  type ClientWire,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
/** One upstream call may take this long before it counts as failed. */
const UPSTREAM_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** Requests served so far: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/**
 * Per-provider failure streaks. A provider that refuses three attempts in a
 * row is treated as spent for the rest of the trial: the generator schedules
 * at most a pair of consecutive failures per provider, so three in a row
 * means its budget is spent or its outage has begun, and retrying it on
 * every later request would only spend tie-break calls. A success resets
 * the streak, and a provider is always retried on the next request after a
 * first or second failure — a single refusal may be its scheduled failure,
 * not a spent budget.
 */
const providerState = new Map<string, { streak: number; dead: boolean }>();

function stateOf(providerId: string): { streak: number; dead: boolean } {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { streak: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models this entry serves, in brief order, one row per provider. */
function servedModels(): { id: string; provider: string; displayName: string }[] {
  const out: { id: string; provider: string; displayName: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      // The catalog is the source of truth for what is free.
      const inCatalog = catalog.providers
        .find((p) => p.id === probe.provider.id)
        ?.models.find((m) => m.id === model.id);
      if (inCatalog === undefined || !inCatalog.free) continue;
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.id });
    }
  }
  return out;
}

/** Requirements of the request now arriving: the manifest's row, else the body. */
function requirementsFor(body: Json, clientWire: ClientWire): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: URL, headers and body in its wire. */
function upstreamCall(candidate: Candidate, clientWire: ClientWire, body: Json, stream: boolean): UpstreamCall {
  const base = joinBase(ENV_URL, candidate.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.provider.auth, candidate.provider.key),
  };
  const model = candidate.model.id;
  if (candidate.provider.wire === 'openai') {
    const upstream =
      clientWire === 'anthropic' ? anthropicToOpenAI(body, model) : { ...body, model };
    if (stream) {
      upstream.stream = true;
      upstream.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (candidate.provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstream = clientWire === 'anthropic' ? { ...body, model } : openAIToAnthropic(body, model);
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  // gemini
  const upstream = clientWire === 'anthropic' ? anthropicToGemini(body) : openAIToGemini(body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(model)}:${action}${stream ? '?alt=sse' : ''}`;
  return { url, headers, body: upstream };
}

/**
 * Serve one client request through the ranked candidates. A 402, 429 or 5xx
 * (and a transport failure) moves on to the next provider; a 2xx answer is
 * translated back into the client's wire and returned. At most one attempt
 * per provider: a refusal belongs to the provider, not the model.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientWire: ClientWire): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const stream = body.stream === true;
  const requires = requirementsFor(body, clientWire);
  const candidates = candidatesFor(probes, brief.ratings, requires).filter(
    (candidate) => !stateOf(candidate.provider.id).dead,
  );
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  const tried = new Set<string>();
  let lastStatus = 503;
  let lastDetail = 'every provider refused this request';
  for (const candidate of candidates) {
    if (tried.has(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    const call = upstreamCall(candidate, clientWire, body, stream);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
    } catch (err) {
      const state = stateOf(candidate.provider.id);
      state.streak += 1;
      if (state.streak >= 3) state.dead = true;
      lastStatus = 502;
      lastDetail = (err as Error).message;
      continue;
    }
    if (!upstream.ok) {
      const state = stateOf(candidate.provider.id);
      state.streak += 1;
      if (state.streak >= 3) state.dead = true;
      lastStatus = upstream.status;
      try {
        lastDetail = (await upstream.text()).slice(0, 300);
      } catch {
        lastDetail = `HTTP ${upstream.status}`;
      }
      continue;
    }
    stateOf(candidate.provider.id).streak = 0;
    const text = await upstream.text();
    const wire = candidate.provider.wire;
    if (stream) {
      // Same wire both sides and OpenAI: the provider's own stream is already
      // in the client's shape; anything else is re-emitted from the collect.
      if (wire === 'openai' && clientWire === 'openai') {
        sendText(res, 200, 'text/event-stream', text);
        return;
      }
      if (wire === 'anthropic' && clientWire === 'anthropic') {
        sendText(res, 200, 'text/event-stream', text);
        return;
      }
      sendText(res, 200, 'text/event-stream', convertStream(wire, clientWire, text, candidate.model.id));
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      lastStatus = 502;
      lastDetail = 'the provider answered a body that is not JSON';
      continue;
    }
    if (!isRecord(parsed)) {
      lastStatus = 502;
      lastDetail = 'the provider answered a JSON value that is not one object';
      continue;
    }
    if (wire === clientWire) {
      sendJson(res, 200, parsed);
      return;
    }
    sendJson(res, 200, convertJson(wire, clientWire, parsed, candidate.model.id));
    return;
  }
  log(`no provider served the request (last: HTTP ${lastStatus} ${lastDetail.slice(0, 120)})`);
  sendError(res, clientWire, 503, 'overloaded_error', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** POST /register with the port and the one catalog; retry briefly if the env is not up yet. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the workload holds ${reply.requests} requests`);
      }
      return;
    } catch (err) {
      lastError = err;
      await sleep(250);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
