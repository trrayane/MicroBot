"use client";
import { useCallback, useEffect, useState } from "react";
import { me, type MeResponse } from "./api";

export function useAuth() {
  const [user, setUser] = useState<MeResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try { setUser(await me()); } catch { setUser(null); } finally { setLoading(false); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return {
    user, loading,
    isAdmin: user?.role === "admin",
    isMember: user?.role === "admin" || user?.role === "member",
    refresh,
    logout: useCallback(() => {
      sessionStorage.removeItem("mc_token");
      sessionStorage.removeItem("mc_admin_key");
      setUser(null);
      window.location.hash = "";
    }, []),
  };
}
