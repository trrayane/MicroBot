"use client";

import * as React from "react";
import { McButton } from "@/components/ui/mc-button";
import { McInput } from "@/components/ui/mc-input";
import { McCard } from "@/components/ui/mc-card";
import { McBadge } from "@/components/ui/mc-badge";
import { McAvatar, McAvatarFallback, McAvatarImage } from "@/components/ui/mc-avatar";
import { McScrollArea } from "@/components/ui/mc-scrollarea";
import { McSkeleton } from "@/components/ui/mc-skeleton";
import { McSeparator } from "@/components/ui/mc-separator";
import { McTooltip, McTooltipTrigger, McTooltipContent } from "@/components/ui/mc-tooltip";
import { McAlert } from "@/components/ui/mc-alert";
import { chat, discordLoginUrl, feedback, type ChatResponse, type Source } from "@/lib/api";
import { useAuth } from "@/lib/use-auth";
import { toast } from "@/components/ui/mc-sonner";
import { Send, Bot, User, ThumbsUp, ThumbsDown, FileText, MessageSquare, Sparkles, LogOut } from "lucide-react";

interface Message {
  id: string; role: "user" | "assistant"; content: string;
  sources?: Source[]; grounded?: boolean;
}

const SUGGESTIONS = ["Comment rejoindre le Micro Club ?", "Quelles sont les règles du club ?", "Quels projets sont sur le GitHub ?"];

function parseOAuthFragment() {
  if (typeof window === "undefined") return;
  const hash = window.location.hash;
  const tokenMatch = hash.match(/token=([^&]+)/);
  const errorMatch = hash.match(/error=([^&]+)/);
  if (tokenMatch) {
    sessionStorage.setItem("mc_token", tokenMatch[1]);
    sessionStorage.setItem("mc_admin", "1");
    window.history.replaceState(null, "", window.location.pathname);
  } else if (errorMatch) {
    toast.error(errorMatch[1] === "not_member" ? "Vous n'êtes pas membre du serveur Discord." : "Connexion annulée.");
    window.history.replaceState(null, "", window.location.pathname);
  }
}

function renderMarkdown(text: string): string {
  let html = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/^#{1,3} (.+)$/gm, "<strong>$1</strong>");
  html = html.replace(/^[-•] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
  html = html.replace(/\n/g, "<br>");
  html = html.replace(/<br><ul>/g, "<ul>").replace(/<\/ul><br>/g, "</ul>");
  html = html.replace(/\[(\d+(?:,\d+)*)\]/g, '<sup class="cit text-primary font-bold">$1</sup>');
  return html;
}

export default function ChatPage() {
  const { user, isAdmin, isMember, refresh, logout } = useAuth();
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [conversationId, setConversationId] = React.useState<string | undefined>();
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => { parseOAuthFragment(); refresh(); }, [refresh]);
  React.useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }); }, [messages]);

  const send = React.useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setInput("");
    setBusy(true);
    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    try {
      const res: ChatResponse = await chat(trimmed, conversationId);
      setConversationId(res.conversation_id);
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "assistant", content: res.answer, sources: res.sources, grounded: res.grounded }]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Erreur");
      setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
      setInput(trimmed);
    } finally { setBusy(false); }
  }, [busy, conversationId]);

  const newConversation = React.useCallback(() => { setMessages([]); setConversationId(undefined); }, []);

  const rateMessage = React.useCallback(async (vote: "up" | "down") => {
    try { await feedback(conversationId || null, vote); toast.success(vote === "up" ? "Merci !" : "Merci, on améliore."); }
    catch { toast.error("Erreur feedback"); }
  }, [conversationId]);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-72 border-r border-border bg-card flex flex-col">
        <div className="p-4 flex items-center gap-3">
          <img src="/logo-icon.jpg" alt="MC" className="w-10 h-10 rounded-lg object-cover" />
          <div>
            <h1 className="font-bold text-lg leading-tight">MC Club RAG</h1>
            <p className="text-xs text-muted-foreground">Assistant IA</p>
          </div>
        </div>
        <McSeparator />
        <div className="p-3">
          <McButton variant="secondary" size="sm" className="w-full" onClick={newConversation}>
            <MessageSquare className="w-4 h-4 mr-2" /> Nouvelle conversation
          </McButton>
        </div>
        <McSeparator />
        <div className="flex-1 overflow-y-auto p-3" />
        <McSeparator />
        <div className="p-3">
          {user && user.avatar ? (
            <div className="flex items-center gap-3">
              <McAvatar>
                <McAvatarImage src={`https://cdn.discordapp.com/avatars/${user.user_id}/${user.avatar}.png`} />
                <McAvatarFallback>{user.display_name?.[0] || "?"}</McAvatarFallback>
              </McAvatar>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{user.display_name}</p>
                <McBadge variant={isAdmin ? "default" : "secondary"} className="text-xs">
                  {isAdmin ? "Admin" : isMember ? "Membre" : "Visiteur"}
                </McBadge>
              </div>
              <McTooltip>
                <McTooltipTrigger>
                  <button onClick={logout} className="text-muted-foreground hover:text-foreground"><LogOut className="w-4 h-4" /></button>
                </McTooltipTrigger>
                <McTooltipContent>Déconnexion</McTooltipContent>
              </McTooltip>
            </div>
          ) : (
            <McButton variant="primary" size="sm" className="w-full" onClick={() => { window.location.href = discordLoginUrl(); }}>
              Continuer avec Discord
            </McButton>
          )}
        </div>
      </aside>

      {/* Chat */}
      <main className="flex-1 flex flex-col min-w-0">
        <McScrollArea className="flex-1">
          <div ref={scrollRef} className="max-w-3xl mx-auto p-6 space-y-6">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6">
                <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
                  <Sparkles className="w-10 h-10 text-primary" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold mb-2">Bonjour{user?.display_name ? `, ${user.display_name}` : ""} !</h2>
                  <p className="text-muted-foreground max-w-md">Je suis l&apos;assistant IA du Micro Club. Posez-moi une question sur les procédures, événements, règles ou projets du club.</p>
                </div>
                <div className="flex flex-wrap gap-2 justify-center">
                  {SUGGESTIONS.map((s) => (<McButton key={s} variant="tertiary" size="sm" onClick={() => send(s)}>{s}</McButton>))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === "user" ? "bg-secondary" : "bg-primary"}`}>
                  {msg.role === "user" ? <User className="w-4 h-4 text-secondary-foreground" /> : <Bot className="w-4 h-4 text-primary-foreground" />}
                </div>
                <div className="max-w-[80%]">
                  <div
                    className={`rounded-lg p-4 text-sm leading-relaxed ${msg.role === "user" ? "bg-secondary text-secondary-foreground" : msg.grounded ? "bg-card border border-border" : "bg-muted"}`}
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                  />
                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 space-y-2">
                      <p className="text-xs font-medium text-muted-foreground">Sources</p>
                      {msg.sources.map((src, i) => (
                        <McCard key={i} className="p-3">
                          <div className="flex items-center gap-2">
                            <FileText className="w-4 h-4 text-primary shrink-0" />
                            <span className="text-sm font-medium truncate">{src.title}</span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{src.snippet}</p>
                        </McCard>
                      ))}
                      <div className="flex gap-2">
                        <McButton variant="tertiary" size="sm" onClick={() => rateMessage("up")}><ThumbsUp className="w-3 h-3 mr-1" /> Utile</McButton>
                        <McButton variant="tertiary" size="sm" onClick={() => rateMessage("down")}><ThumbsDown className="w-3 h-3 mr-1" /> Pas utile</McButton>
                      </div>
                    </div>
                  )}
                  {msg.role === "assistant" && (!msg.grounded || !msg.sources?.length) && (
                    <McAlert variant="default" title="Réponse non sourcée" description="Vérifiez les informations importantes." className="mt-2" />
                  )}
                </div>
              </div>
            ))}

            {busy && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center"><Bot className="w-4 h-4 text-primary-foreground" /></div>
                <div className="space-y-2"><McSkeleton className="h-4 w-48" /><McSkeleton className="h-4 w-32" /></div>
              </div>
            )}
          </div>
        </McScrollArea>

        <div className="border-t border-border p-4 bg-card">
          <form className="max-w-3xl mx-auto flex gap-2" onSubmit={(e) => { e.preventDefault(); send(input); }}>
            <McInput value={input} onChange={(e) => setInput(e.target.value)} placeholder="Posez votre question..." disabled={busy} className="flex-1" maxLength={2000} />
            <McButton type="submit" variant="primary" size="md" disabled={busy || !input.trim()}><Send className="w-4 h-4" /></McButton>
          </form>
        </div>
      </main>
    </div>
  );
}
