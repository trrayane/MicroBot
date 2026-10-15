import type { Metadata } from "next";
import { DM_Sans, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import McSonner from "@/components/ui/mc-sonner";

const dmSans = DM_Sans({ subsets: ["latin"], variable: "--font-dm-sans" });
const jakarta = Plus_Jakarta_Sans({ subsets: ["latin"], variable: "--font-plus-jakarta-sans" });

export const metadata: Metadata = {
  title: "MC Club RAG — Assistant",
  description: "Assistant IA du Micro Club",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="fr" className={cn("h-full antialiased", dmSans.variable, jakarta.variable)}>
      <body className="min-h-full flex flex-col bg-background text-foreground font-sans">
        {children}
        <McSonner />
      </body>
    </html>
  );
}
