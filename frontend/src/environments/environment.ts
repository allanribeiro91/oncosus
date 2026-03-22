/**
 * URL direta evita falhas do proxy do `ng serve` com alguns stacks (ex.: HttpClient + fetch).
 * CORS em app.py já permite 127.0.0.1:4200/4201 e regex de portas locais.
 */
export const environment = {
  production: false,
  apiUrl: 'http://127.0.0.1:8000/api',
};
