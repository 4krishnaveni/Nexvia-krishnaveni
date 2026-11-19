const BASE = "/api";
let token = sessionStorage.getItem("token") || "";

export const isLoggedIn = () => !!token;
export function logout() {
  token = "";
  sessionStorage.removeItem("token");
}

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, {
    ...opts,
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (res.status === 401 || res.status === 403) {
    logout();
    throw new Error("auth");
  }
  if (!res.ok) throw new Error((await res.text()) || res.statusText);
  return res.json();
}

export async function login(username, password) {
  const d = await req("/auth/login", { method: "POST", body: JSON.stringify({ username, password }) });
  token = d.access_token;
  sessionStorage.setItem("token", token);
}
export const listStudies = () => req("/studies");
export const decide = (id, decision, comment) =>
  req(`/studies/${id}/decision`, { method: "POST", body: JSON.stringify({ decision, comment }) });
export const getAudit = () => req("/audit");
