import { readFileSync } from 'node:fs'

const settings = readFileSync(new URL('../src/pages/SettingsPage.tsx', import.meta.url), 'utf8')
const consultation = readFileSync(
  new URL('../src/components/consultation/ConsultationForm.tsx', import.meta.url),
  'utf8',
)
const api = readFileSync(new URL('../src/lib/api.ts', import.meta.url), 'utf8')

const checks = [
  ['typed LLM credential scope prop', settings.includes("credentialScope: 'consultation' | 'knowledge' | 'vision'")],
  ['consultation scope call site', settings.includes('credentialScope="consultation"')],
  ['knowledge scope call site', settings.includes('credentialScope="knowledge"')],
  ['vision scope call site', settings.includes('credentialScope="vision"')],
  ['scope sent to models endpoint', settings.includes('credential_scope: credentialScope')],
  ['scope sent to text test endpoint', settings.includes("credential_scope: credentialScope === 'knowledge' ? 'knowledge' : 'consultation'")],
  ['typed models API scope', api.includes("credential_scope: 'consultation' | 'knowledge' | 'vision'")],
  ['model request generation exists', settings.includes('modelRequestGeneration = useRef(0)')],
  ['model generation advances per request', settings.includes('++modelRequestGeneration.current')],
  [
    'stale model responses are ignored',
    settings.includes('requestGeneration !== modelRequestGeneration.current'),
  ],
  [
    'provider change normalizes Base URL',
    settings.includes("targetProvider?.needs_base_url ? baseUrl : ''"),
  ],
  ['normalized Base URL is stored', settings.includes('setBaseUrl(normalizedBaseUrl)')],
  ['normalized Base URL is used for requests', settings.includes('base_url: normalizedBaseUrl || null')],
  ['LLM provider change clears API key', settings.includes("setApiKey('')")],
  ['provider refresh does not reuse prior key', settings.includes("fetchModels(p, '', normalizedBaseUrl)")],
  ['embedding provider refresh does not reuse prior key', settings.includes("fetchModels(p, '', normalizedBaseUrl)") && settings.match(/setApiKey\(''\)/g)?.length >= 2],
  ['consultation accepts WebP', consultation.includes('.webp')],
  ['consultation accepts GIF', consultation.includes('.gif')],
  ['consultation copy mentions WebP', consultation.includes('WEBP')],
  ['consultation copy mentions GIF', consultation.includes('GIF')],
]

const failed = checks.filter(([, passed]) => !passed)
if (failed.length > 0) {
  for (const [name] of failed) console.error(`Missing contract: ${name}`)
  process.exit(1)
}

console.log(`${checks.length} final-review frontend contracts passed`)
