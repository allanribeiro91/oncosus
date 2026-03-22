/**
 * Em `npm start`, o proxy (proxy.conf.json) encaminha /api → http://127.0.0.1:8000
 * e evita CORS (mesma origem: 127.0.0.1:4200).
 */
export const environment = {
  production: false,
  apiUrl: '/api',
};
