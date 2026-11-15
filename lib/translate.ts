// translate.ts — the three provider wires, both client wires.
//
// The trial's providers answer with an echo of what they parsed: the system
// prompt, the user text, the images, the tool names, the stop sequences, the
// max_tokens and the JSON mode. The client compares that echo with what it
// sent, so a translation that rewrites or drops any of those fields loses the
// request — every converter below carries all of them, in order, untouched.
// Usage counts are carried verbatim from the provider that answered, never
// recomputed: the client checks them against the provider's own record.
// Stop reasons map across wires by family (stop/end_turn/STOP,
// length/max_tokens/MAX_TOKENS, tool_calls/tool_use, content_filter/refusal).
//
// Streams are collected whole and then re-emitted in the client's wire: the
// providers answer locally and fast, and re-emitting from the collected
// answer is the only way to put the usage into Anthropic's message_start
// (which comes first) and OpenAI's final usage chunk (which comes last)
// with the provider's exact counts.

import { isRecord, type Json } from './brief.ts';

/** Fixed epoch second for every completion's `created`: no clock is read. */
const FIXED_CREATED = 1767225600;
/** Fixed instant for every Anthropic listing's `created_at`. */
export const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export type ClientWire = 'openai' | 'anthropic';

// --- small helpers -----------------------------------------------------------

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function safeId(model: string): string {
  return model.replace(/[^A-Za-z0-9._-]/g, '-');
}

function splitPieces(text: string, size = 64): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One Anthropic content block as an OpenAI content part, or null. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** One OpenAI content part as an Anthropic content block, or null. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return {
        type: 'image',
        source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] },
      };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  if (part.type === 'image' && isRecord(part.source)) {
    // Already Anthropic-shaped (a client that sent one through by mistake).
    return part;
  }
  return null;
}

/** One OpenAI/Anthropic image as a Gemini part, or null. */
function geminiPartOfImage(url: string): Json | null {
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  }
  if (url !== '') return { fileData: { fileUri: url } };
  return null;
}

function geminiPartsOfOpenAIContent(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const part = geminiPartOfImage(block.image_url.url);
      if (part !== null) parts.push(part);
    }
  }
  return parts;
}

function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

// --- stop reasons ------------------------------------------------------------

export function openAIFinishOfAnthropic(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

export function anthropicStopOfOpenAI(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

export function openAIFinishOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

export function anthropicStopOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'refusal';
  return 'end_turn';
}

// --- request builders: client -> upstream ------------------------------------

/** Anthropic client -> OpenAI upstream. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI client -> Anthropic upstream. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof content === 'string' ? content : toolResultText(content),
          },
        ],
      });
      continue;
    }
    // user (or anything else treated as user)
    const blocks: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = anthropicBlockOf(part);
        if (block !== null) blocks.push(block);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

function geminiConfigFromOpenAI(body: Json): Json {
  const config: Json = {};
  const maxTokens =
    typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  return config;
}

function geminiToolsFromOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** OpenAI client -> Gemini upstream. */
export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          text: `call ${typeof call.function.name === 'string' ? call.function.name : ''}(${typeof call.function.arguments === 'string' ? call.function.arguments : ''})`,
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : toolResultText(raw.content);
      contents.push({ role: 'user', parts: [{ text }] });
      continue;
    }
    const parts = geminiPartsOfOpenAIContent(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiToolsFromOpenAI(body);
  if (tools.length > 0) out.tools = tools;
  const config = geminiConfigFromOpenAI(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Anthropic client -> Gemini upstream. */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({
              text: `call ${typeof block.name === 'string' ? block.name : ''}(${JSON.stringify(isRecord(block.input) ? block.input : {})})`,
            });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({
              inlineData: {
                mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
                data: block.source.data,
              },
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            const part = geminiPartOfImage(block.source.url);
            if (part !== null) parts.push(part);
          }
        } else if (block.type === 'tool_result') {
          parts.push({ text: toolResultText(block.content) });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// --- response builders: upstream JSON -> client JSON -------------------------

export interface Usage {
  input: number;
  output: number;
}

export interface CollectedText {
  text: string;
  /** OpenAI finish_reason / Anthropic stop_reason / Gemini finishReason, raw. */
  stop: string | null;
  usage: Usage | null;
  toolCalls: { id: string; name: string; arguments: string }[];
}

function openAICompletion(model: string, collected: CollectedText): Json {
  const message: Json = { role: 'assistant', content: collected.text };
  if (collected.toolCalls.length > 0) {
    message.tool_calls = collected.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
    if (collected.text === '') message.content = null;
  }
  const usage = collected.usage ?? { input: 0, output: 0 };
  return {
    id: `chatcmpl-${safeId(model)}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: collected.stop ?? 'stop',
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

function anthropicMessage(model: string, upstreamId: string, collected: CollectedText): Json {
  const content: Json[] = [];
  if (collected.text !== '') content.push({ type: 'text', text: collected.text });
  for (const call of collected.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = collected.usage ?? { input: 0, output: 0 };
  return {
    id: `msg_${upstreamId}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: collected.stop ?? 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** Parse an upstream JSON answer into text + stop + usage, whatever its wire. */
export function collectJson(wire: 'openai' | 'anthropic' | 'gemini', body: Json): CollectedText {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const toolCalls = (Array.isArray(message.tool_calls) ? message.tool_calls : []).flatMap((call) => {
      if (!isRecord(call) || !isRecord(call.function)) return [];
      return [
        {
          id: typeof call.id === 'string' ? call.id : 'call_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
        },
      ];
    });
    const usage = isRecord(body.usage)
      ? { input: numOrZero(body.usage.prompt_tokens), output: numOrZero(body.usage.completion_tokens) }
      : null;
    return {
      text,
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
      usage,
      toolCalls,
    };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(body.content) ? body.content : [];
    const text = content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
    const toolCalls = content.flatMap((b) => {
      if (!isRecord(b) || b.type !== 'tool_use') return [];
      return [
        {
          id: typeof b.id === 'string' ? b.id : 'toolu_unknown',
          name: typeof b.name === 'string' ? b.name : '',
          arguments: JSON.stringify(isRecord(b.input) ? b.input : {}),
        },
      ];
    });
    const usage = isRecord(body.usage)
      ? { input: numOrZero(body.usage.input_tokens), output: numOrZero(body.usage.output_tokens) }
      : null;
    return {
      text,
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
      usage,
      toolCalls,
    };
  }
  // gemini
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const toolCalls = parts.flatMap((p) => {
    if (!isRecord(p) || !isRecord(p.functionCall)) return [];
    return [
      {
        id: `call_${typeof p.functionCall.name === 'string' ? safeId(p.functionCall.name) : 'unknown'}`,
        name: typeof p.functionCall.name === 'string' ? p.functionCall.name : '',
        arguments: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}),
      },
    ];
  });
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const usage = {
    input: numOrZero(meta.promptTokenCount),
    output: numOrZero(meta.candidatesTokenCount),
  };
  return {
    text,
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    usage,
    toolCalls,
  };
}

export function upstreamIdOf(wire: 'openai' | 'anthropic' | 'gemini', body: Json): string {
  if (wire === 'openai' && typeof body.id === 'string') return body.id;
  if (wire === 'anthropic' && typeof body.id === 'string') return body.id;
  if (wire === 'gemini' && typeof body.modelVersion === 'string') return body.modelVersion;
  return 'aggregate';
}

/** An upstream JSON answer rewritten into the client's wire. */
export function convertJson(
  upstreamWire: 'openai' | 'anthropic' | 'gemini',
  clientWire: ClientWire,
  body: Json,
  model: string,
): Json {
  const collected = collectJson(upstreamWire, body);
  if (clientWire === 'openai') {
    // Map the stop reason into OpenAI's family.
    const finish =
      upstreamWire === 'anthropic'
        ? openAIFinishOfAnthropic(collected.stop)
        : upstreamWire === 'gemini'
          ? openAIFinishOfGemini(collected.stop)
          : (collected.stop ?? 'stop');
    return openAICompletion(model, { ...collected, stop: finish });
  }
  const stop =
    upstreamWire === 'openai'
      ? anthropicStopOfOpenAI(collected.stop)
      : upstreamWire === 'gemini'
        ? anthropicStopOfGemini(collected.stop)
        : (collected.stop ?? 'end_turn');
  return anthropicMessage(model, upstreamIdOf(upstreamWire, body), { ...collected, stop });
}

// --- SSE parsing -------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(data: string): Json | null {
  try {
    const value: unknown = JSON.parse(data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Collect an OpenAI SSE stream. */
export function collectOpenAIStream(text: string): CollectedText {
  let out = '';
  let finish: string | null = null;
  let usage: Usage | null = null;
  const toolCalls: CollectedText['toolCalls'] = [];
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numOrZero(chunk.usage.prompt_tokens), output: numOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    for (const call of Array.isArray(delta.tool_calls) ? delta.tool_calls : []) {
      if (!isRecord(call)) continue;
      const fn = isRecord(call.function) ? call.function : {};
      const index = typeof call.index === 'number' ? call.index : toolCalls.length;
      while (toolCalls.length <= index) toolCalls.push({ id: '', name: '', arguments: '' });
      const slot = toolCalls[index];
      if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') slot.name = fn.name;
      if (typeof fn.arguments === 'string') slot.arguments += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  return { text: out, stop: finish ?? 'stop', usage, toolCalls: toolCalls.filter((c) => c.name !== '' || c.id !== '') };
}

/** Collect an Anthropic SSE stream. */
export function collectAnthropicStream(text: string): CollectedText {
  let out = '';
  let stop: string | null = null;
  let input = 0;
  let output = 0;
  const toolCalls: CollectedText['toolCalls'] = [];
  let currentTool: { id: string; name: string; arguments: string } | null = null;
  for (const frame of parseSse(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const block = data.content_block;
      if (block.type === 'tool_use') {
        currentTool = {
          id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
          name: typeof block.name === 'string' ? block.name : '',
          arguments: '',
        };
        toolCalls.push(currentTool);
      }
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && currentTool !== null) {
        currentTool.arguments += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, stop: stop ?? 'end_turn', usage: { input, output }, toolCalls };
}

/** Collect a Gemini SSE stream. */
export function collectGeminiStream(text: string): CollectedText {
  let out = '';
  let finish: string | null = null;
  let usage: Usage | null = null;
  for (const frame of parseSse(text)) {
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') out += part.text;
      }
      if (typeof candidate.finishReason === 'string') finish = candidate.finishReason;
    }
    if (isRecord(chunk.usageMetadata)) {
      usage = {
        input: numOrZero(chunk.usageMetadata.promptTokenCount),
        output: numOrZero(chunk.usageMetadata.candidatesTokenCount),
      };
    }
  }
  return { text: out, stop: finish ?? 'STOP', usage, toolCalls: [] };
}

export function collectStream(wire: 'openai' | 'anthropic' | 'gemini', text: string): CollectedText {
  if (wire === 'openai') return collectOpenAIStream(text);
  if (wire === 'anthropic') return collectAnthropicStream(text);
  return collectGeminiStream(text);
}

// --- SSE emission ------------------------------------------------------------

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Emit a collected answer as an OpenAI SSE stream (with usage chunk + [DONE]). */
export function emitOpenAIStream(model: string, collected: CollectedText): string {
  const usage = collected.usage ?? { input: 0, output: 0 };
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: `chatcmpl-${safeId(model)}`,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitPieces(collected.text)) out += sseEvent(null, chunk({ content: piece }, null));
  for (const call of collected.toolCalls) {
    out += sseEvent(
      null,
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: call.id,
              type: 'function',
              function: { name: call.name, arguments: call.arguments },
            },
          ],
        },
        null,
      ),
    );
  }
  out += sseEvent(null, chunk({}, collected.stop ?? 'stop'));
  out += sseEvent(null, {
    id: `chatcmpl-${safeId(model)}`,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** Emit a collected answer as an Anthropic SSE stream, in canonical event order. */
export function emitAnthropicStream(model: string, collected: CollectedText): string {
  const usage = collected.usage ?? { input: 0, output: 0 };
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: `msg_${safeId(model)}`,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: 0 },
    },
  });
  let index = 0;
  if (collected.text !== '' || collected.toolCalls.length === 0) {
    out += sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'text', text: '' },
    });
    for (const piece of splitPieces(collected.text)) {
      out += sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: piece },
      });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of collected.toolCalls) {
    out += sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
    });
    if (call.arguments !== '') {
      out += sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: call.arguments },
      });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: collected.stop ?? 'end_turn', stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

/** A collected upstream stream re-emitted in the client's wire. */
export function convertStream(
  upstreamWire: 'openai' | 'anthropic' | 'gemini',
  clientWire: ClientWire,
  text: string,
  model: string,
): string {
  const collected = collectStream(upstreamWire, text);
  if (clientWire === 'openai') {
    const finish =
      upstreamWire === 'anthropic'
        ? openAIFinishOfAnthropic(collected.stop)
        : upstreamWire === 'gemini'
          ? openAIFinishOfGemini(collected.stop)
          : (collected.stop ?? 'stop');
    return emitOpenAIStream(model, { ...collected, stop: finish });
  }
  const stop =
    upstreamWire === 'openai'
      ? anthropicStopOfOpenAI(collected.stop)
      : upstreamWire === 'gemini'
        ? anthropicStopOfGemini(collected.stop)
        : (collected.stop ?? 'end_turn');
  return emitAnthropicStream(model, { ...collected, stop });
}

// --- model listings ----------------------------------------------------------

/** GET /v1/models, OpenAI shape: every free model id this entry serves. */
export function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models, Anthropic shape. */
export function anthropicModelList(models: readonly { id: string; displayName: string }[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.displayName,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
