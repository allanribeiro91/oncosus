import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, TimeoutError, catchError, throwError, timeout } from 'rxjs';
import { environment } from '../../../environments/environment';

/** RAG + Ollama em CPU pode levar vários minutos na primeira resposta (carregar modelo). */
const CHAT_TIMEOUT_MS = 300_000;

export interface ChatResponse {
  question: string;
  answer: string;
  sources: string[];
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  ask(question: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, { question }).pipe(
      timeout(CHAT_TIMEOUT_MS),
      catchError((err: unknown) => {
        if (err instanceof TimeoutError) {
          return throwError(() => ({
            error: {
              detail:
                `Tempo esgotado (${CHAT_TIMEOUT_MS / 60_000} min). O assistente usa Ollama local — na primeira pergunta o modelo pode demorar muito em CPU ou ficar sem RAM. Confira se o backend está em http://127.0.0.1:8000 e o Ollama ativo; veja também .env (ONCOSUS_OLLAMA_MODEL, ONCOSUS_OLLAMA_NUM_CTX).`,
            },
          }));
        }
        return throwError(() => err);
      }),
    );
  }
}
