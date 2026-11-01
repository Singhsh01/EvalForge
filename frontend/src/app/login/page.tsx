"use client";

import { Button, ErrorNote, Field, inputCls } from "@/components/ui";
import { api, setSession } from "@/lib/api";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      const r = await api.login(username, password);
      setSession(r.token, r.user);
      router.replace("/");
    } catch (err) { setError(err); } finally { setBusy(false); }
  }

  return (
    <div className="grid min-h-dvh place-items-center px-4">
      <form onSubmit={submit} className="glass w-full max-w-sm space-y-5 rounded-2xl p-7">
        <div>
          <h1 className="text-2xl font-semibold text-white">Sign in to EvalForge</h1>
          <p className="mt-1 text-sm text-white/50">Roles decide what you can change: viewers read, engineers build, approvers approve production changes.</p>
        </div>
        <Field label="Username"><input className={inputCls} value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required /></Field>
        <Field label="Password"><input className={inputCls} type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required /></Field>
        <ErrorNote error={error} />
        <Button type="submit" className="w-full" loading={busy}>Sign in</Button>
        <p className="text-xs leading-relaxed text-white/40">Local development creates <b>admin</b>, <b>engineer</b>, <b>approver</b> and <b>viewer</b> with the password from <code>EVALFORGE_BOOTSTRAP_PASSWORD</code> (default <code>evalforge-dev</code>).</p>
      </form>
    </div>
  );
}
