import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TimeoutError, finalize } from 'rxjs';
import { ApiHealth, ChatService, httpErrorDetail } from '../../core/services/chat.service';

interface Message {
  question: string;
  answer: string;
  sources: string[];
  isUser: boolean;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
})
export class ChatComponent implements OnInit {
  question = '';
  loading = false;
  error: string | null = null;
  messages: Message[] = [];
  /** Banner fixo: conexão / RAG / Ollama antes de enviar pergunta. */
  healthSeverity: 'ok' | 'warn' | 'error' = 'ok';
  healthMessage: string | null = null;

  constructor(private chatService: ChatService) {}

  ngOnInit(): void {
    this.chatService.health().subscribe({
      next: (h: ApiHealth) => this.applyHealth(h),
      error: (err: unknown) => {
        this.healthSeverity = 'error';
        this.healthMessage =
          err instanceof TimeoutError
            ? 'API não respondeu em http://127.0.0.1:8000. No PowerShell: cd backend\\rag e .\\start-api.ps1'
            : httpErrorDetail(err);
      },
    });
  }

  private applyHealth(h: ApiHealth): void {
    if (h.status === 'unavailable') {
      this.healthSeverity = 'error';
      this.healthMessage =
        h.startup_error?.trim() ||
        'RAG não inicializou (vectorstore ou erro 1455 ao carregar embeddings). O chat vai falhar até corrigir o backend.';
      return;
    }
    if (h.status === 'degraded') {
      this.healthSeverity = 'warn';
      this.healthMessage = `RAG ok, mas Ollama com problema: ${String(h.ollama)}`;
      return;
    }
    this.healthSeverity = 'ok';
    this.healthMessage = null;
  }

  send(): void {
    const q = this.question.trim();
    if (!q || this.loading) return;

    this.error = null;
    this.messages.push({ question: q, answer: '', sources: [], isUser: true });
    this.question = '';
    this.loading = true;

    this.chatService
      .ask(q)
      .pipe(finalize(() => (this.loading = false)))
      .subscribe({
        next: (res) => {
          this.messages.push({
            question: res.question,
            answer: res.answer,
            sources: res.sources,
            isUser: false,
          });
        },
        error: (err: unknown) => {
          this.error = httpErrorDetail(err);
        },
      });
  }
}
