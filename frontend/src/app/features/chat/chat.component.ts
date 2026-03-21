import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../core/services/chat.service';

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
export class ChatComponent {
  question = '';
  loading = false;
  error: string | null = null;
  messages: Message[] = [];

  constructor(private chatService: ChatService) {}

  send(): void {
    const q = this.question.trim();
    if (!q || this.loading) return;

    this.error = null;
    this.messages.push({ question: q, answer: '', sources: [], isUser: true });
    this.question = '';
    this.loading = true;

    this.chatService.ask(q).subscribe({
      next: (res) => {
        this.messages.push({
          question: res.question,
          answer: res.answer,
          sources: res.sources,
          isUser: false,
        });
        this.loading = false;
      },
      error: (err) => {
        this.error = err?.error?.detail || err?.message || 'Erro ao conectar com a API. Verifique se o backend está rodando.';
        this.loading = false;
      },
    });
  }
}
