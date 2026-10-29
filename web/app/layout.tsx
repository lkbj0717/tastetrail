import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TasteTrail · Find the film that fits",
  description: "Compare GLM, a preference Agent, and Agent + Qloo using your movie taste.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
