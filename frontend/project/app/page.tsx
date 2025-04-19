"use client";

import {
  Heart,
  Shield,
  Brain,
  AlertCircle,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { FlickeringGrid } from "@/components/magicui/flickering-grid";
import { InteractiveHoverButton } from "@/components/magicui/interactive-hover-button";

export default function Home() {
  const router = useRouter();

  return (
    <main className="relative min-h-screen bg-gradient-to-b from-background to-secondary overflow-hidden">
      <div className="absolute inset-0">
        <FlickeringGrid
          className="w-full h-full"
          squareSize={4}
          gridGap={6}
          color="#6B7280"
          maxOpacity={0.3}
          flickerChance={0.3}
        />
      </div>
      <div className="container relative z-10 mx-auto px-4 py-16 md:py-24">
        <div className="flex flex-col items-center justify-center space-y-12 text-center">
          <div className="space-y-4">
            <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl">
              SeriniBot
              <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl">
                Your Caring
                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  {" "}
                  Mental Health{" "}
                </span>
                Companion
              </h1>
            </h1>
            <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
              A compassionate AI chatbot designed to help detect signs of
              depression and provide initial mental health support. Talk to us
              in confidence and take the first step towards understanding your
              mental well-being.
            </p>
          </div>

          <InteractiveHoverButton onClick={() => router.push("/chat")}>Start Assesment</InteractiveHoverButton>
          

          <div className="flex flex-col items-center space-y-0">
            <p className="text-sm text-muted-foreground">
              100% Private • Anonymous • Free Initial Assessment
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            {features.map((Feature, index) => (
              <div
                key={index}
                className="rounded-lg border bg-card/80 p-6 shadow-lg transition-all hover:shadow-xl text-center backdrop-blur-sm"
              >
                <div className="flex justify-center">
                  <Feature.icon
                    className="mb-4 h-12 w-12 text-primary"
                    aria-hidden="true"
                  />
                </div>
                <h3 className="mb-2 text-xl font-semibold">{Feature.title}</h3>
                <p className="text-muted-foreground">{Feature.description}</p>
              </div>
            ))}
          </div>

          <div className="mx-auto max-w-[700px] rounded-lg border bg-card/80 p-6 text-left backdrop-blur-sm">
            <div className="flex justify-center mb-4">
              <AlertCircle
                className="h-6 w-6 text-yellow-500"
                aria-hidden="true"
              />
            </div>
            <h4 className="mb-2 text-lg font-semibold text-center">
              Important Note
            </h4>
            <p className="text-sm text-muted-foreground">
              This AI tool is designed for initial screening and support only.
              It is not a substitute for professional medical advice, diagnosis,
              or treatment. If you're experiencing severe symptoms or having
              thoughts of self-harm, please contact emergency services or speak
              with a qualified mental health professional immediately.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}

const features = [
  {
    icon: Brain,
    title: "Depression Screening",
    description:
      "Engage in natural conversations that help identify potential signs of depression using clinically-informed assessment patterns.",
  },
  {
    icon: Shield,
    title: "Private & Secure",
    description:
      "Your conversations are completely private and anonymous. We prioritize your confidentiality and data security.",
  },
  {
    icon: Heart,
    title: "Compassionate Support",
    description:
      "Receive empathetic responses and helpful resources tailored to your unique situation and emotional state.",
  },
];
