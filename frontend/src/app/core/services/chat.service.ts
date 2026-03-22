import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, TimeoutError, catchError, throwError, timeout } from 'rxjs';
import { environment } from '../../../environments/environment';

/** Deve ser um pouco maior que ONCOSUS_OLLAMA_TIMEOUT_SEC no backend (padrão 660s). */
const CHAT_TIMEOUT_MS = 720_000;
const HEALTH_TIMEOUT_MS = 12_000;

export interface ChatResponse {
  question: string;
  answer: string;
  sources: string[];
}

export interface ApiHealth {
  status: string;
  vectorstore_path?: string;
  vectorstore_exists?: boolean;
  ollama_model?: string;
  startup_error?: string;
  ollama?: string;
}

export function httpErrorDetail(err: unknown): string {
  if (err instanceof TimeoutError) {
    return `Tempo esgotado (${CHAT_TIMEOUT_MS / 60_000} min). O assistente usa Ollama na CPU — aqueça o modelo com: ollama run <seu-modelo> "oi". Confira Ollama na bandeja, backend em http://127.0.0.1:8000 e .env: ONCOSUS_OLLAMA_MODEL, ONCOSUS_OLLAMA_TIMEOUT_SEC, ONCOSUS_OLLAMA_NUM_PREDICT (limite de tokens).`;
  }
  if (err instanceof HttpErrorResponse) {
    if (err.status === 0) {
      return 'Sem conexão com a API em http://127.0.0.1:8000. Suba o backend (start-api.ps1) e confira firewall/antivírus.';
    }
    const body = err.error;
    if (body && typeof body === 'object' && 'detail' in body) {
      const d = (body as { detail: unknown }).detail;
      if (typeof d === 'string') {
        return d;
      }
      if (Array.isArray(d)) {
        return d.map((x) => (typeof x === 'object' && x && 'msg' in x ? String((x as { msg: string }).msg) : JSON.stringify(x))).join('; ');
      }
    }
    if (typeof body === 'string' && body.trim()) {
      return body;
    }
    return err.message || `Erro HTTP ${err.status}`;
  }
  if (typeof err === 'object' && err !== null && 'error' in err) {
    const w = (err as { error?: { detail?: string } }).error?.detail;
    if (w) {
      return w;
    }
  }
  return 'Erro ao conectar com a API.';
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  health(): Observable<ApiHealth> {
    return this.http.get<ApiHealth>(`${this.apiUrl}/health`).pipe(timeout(HEALTH_TIMEOUT_MS));
  }

  ask(question: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, { question }).pipe(
      timeout(CHAT_TIMEOUT_MS),
      catchError((err: unknown) => {
        if (err instanceof TimeoutError) {
          return throwError(() => ({
            error: {
              detail: httpErrorDetail(err),
            },
          }));
        }
        return throwError(() => err);
      }),
    );
  }
}
