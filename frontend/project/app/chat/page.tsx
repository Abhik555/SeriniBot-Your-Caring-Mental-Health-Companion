"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Bot, User, ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import { FlickeringGrid } from "@/components/magicui/flickering-grid";

interface Message {
  id: number;
  text: string;
  isBot: boolean;
}

interface UserResponse {
  questionId: number;
  question: string;
  answer: string;
}

// Complete pool of 50 open-ended depression screening questions
const questionPool = [
  "Can you describe your mood over the past two weeks?",
  "What are some thoughts that have been weighing on your mind recently?",
  "How do you feel about your energy levels on an average day?",
  "In what ways have you noticed changes in your sleep patterns?",
  "What has your appetite been like lately, and how do you feel about it?",
  "Could you describe any recent changes in your weight or eating habits?",
  "How do you perceive your interest or pleasure in activities that you used to enjoy?",
  "Can you tell me about any feelings of sadness or hopelessness you've experienced?",
  "What kinds of things do you think might be contributing to your mood right now?",
  "How would you describe your level of concentration on daily tasks?",
  "What challenges have you faced in your daily routine that might be affecting your mood?",
  "Can you share some experiences that have made you feel overwhelmed recently?",
  "How do you deal with negative thoughts or emotions when they arise?",
  "Describe any changes you've noticed in your motivation levels.",
  "How do you feel about your social interactions lately?",
  "What does a typical day look like for you, emotionally speaking?",
  "Can you elaborate on any feelings of guilt or self-criticism you've experienced?",
  "How do you experience stress in your daily life?",
  "What coping strategies do you use when you feel low?",
  "How have you been feeling about your personal relationships recently?",
  "Can you describe how your mood affects your ability to perform daily tasks?",
  "What are some emotions you have been struggling to manage?",
  "How do you describe your level of interest in activities that you once found enjoyable?",
  "What changes have you noticed in your behavior when you feel down?",
  "Can you explain how you feel when you think about your future?",
  "What experiences have contributed to your current emotional state?",
  "How do you express your feelings when you're upset or anxious?",
  "What kind of thoughts go through your mind when you wake up in the morning?",
  "Can you share a recent experience that made you feel particularly low?",
  "How do you perceive your self-worth and confidence on an average day?",
  "What do you feel when you reflect on your personal achievements?",
  "Can you discuss any challenges you face in maintaining a positive outlook?",
  "How do you cope with feelings of isolation or loneliness?",
  "What role do your daily routines play in your overall mood?",
  "How do you handle moments when you feel emotionally drained?",
  "What are some personal goals you have that feel out of reach right now?",
  "Can you describe any times when you felt overwhelmed by your emotions?",
  "How do you navigate days when everything feels harder than usual?",
  "What are some triggers that you have noticed for your low mood?",
  "Can you share how you respond emotionally to stressful situations?",
  "How do you feel about your ability to manage your emotions effectively?",
  "What changes in your lifestyle do you think might improve your mood?",
  "How do you describe your overall sense of well-being?",
  "Can you explain any physical sensations that accompany your low mood?",
  "What do you find most challenging about your emotional experiences lately?",
  "How do you feel your emotional state affects your relationships?",
  "Can you share a time when you felt a significant shift in your mood?",
  "How would you describe the impact of your mood on your daily activities?",
  "What steps have you taken, if any, to improve your emotional health?",
  "How do you envision your path to feeling better or more balanced?",
];

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm here to help assess how you're feeling. I'll ask you a series of questions to better understand your situation. Please answer honestly - there are no right or wrong answers. Shall we begin? type yes to begin.",
      isBot: true,
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(-1);
  const [selectedQuestions, setSelectedQuestions] = useState<string[]>([]);
  const [userResponses, setUserResponses] = useState<UserResponse[]>([]);
  const [conversationComplete, setConversationComplete] = useState(false);
  const router = useRouter();
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the bottom on every new message
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Shuffle questions and pick 10 at random when component mounts
  useEffect(() => {
    const shuffled = [...questionPool].sort(() => 0.5 - Math.random());
    setSelectedQuestions(shuffled.slice(0, 10));
  }, []);

  const handleSend = () => {
    // Do not allow input if conversation is complete
    if (conversationComplete) return;

    const userInput = input.trim();
    if (!userInput) return;

    // Add user's message
    const userMessage: Message = {
      id: messages.length + 1,
      text: userInput,
      isBot: false,
    };
    setMessages((prev) => [...prev, userMessage]);

    // Record the response if a question is active
    if (
      currentQuestionIndex >= 0 &&
      currentQuestionIndex < selectedQuestions.length
    ) {
      setUserResponses((prev) => [
        ...prev,
        {
          questionId: currentQuestionIndex + 1,
          question: selectedQuestions[currentQuestionIndex],
          answer: userInput,
        },
      ]);
    }

    setInput("");
    setIsTyping(true);

    setTimeout(() => {
      let nextMessage = "";
      // Starting conversation: ask the first question
      if (currentQuestionIndex === -1) {
        nextMessage = selectedQuestions[0];
        setCurrentQuestionIndex(0);
      }
      // More questions remain: ask the next question
      else if (currentQuestionIndex < selectedQuestions.length - 1) {
        nextMessage = selectedQuestions[currentQuestionIndex + 1];
        setCurrentQuestionIndex(currentQuestionIndex + 1);
      }
      // Last question answered: begin evaluation
      else {
        nextMessage = "Evaluating your responses";
        setCurrentQuestionIndex(selectedQuestions.length);

        // Combine all user responses into one string.
        const combinedText = [
          ...userResponses.map((resp) => resp.answer),
          userInput,
        ].join(" ");
        console.log(combinedText);

        // Call the backend using POST at /predict; note that the backend might be slow.
        fetch("http://localhost:8000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: combinedText }),
        })
          .then((response) => response.json())
          .then((data) => {
            // data is expected to be a boolean (true or false)
            //console.log(data);
            //console.log(data['result'])
            let resultMessage = "";
            if (data["result"] === true) {
              resultMessage =
                "I'm truly sorry to hear that you might be experiencing depression. Your responses suggest that you're going through a difficult time. Please consider reaching out to someone who can help. Here are some resources you might find useful: [National Suicide Prevention Lifeline]: https://icallhelpline.org/ or call 9152987821 for immediate support. It might also help to talk to a trusted friend, family member, or mental health professional. Remember, you're not alone and help is available.";
            } else {
              resultMessage =
                "That's wonderful to hear! Your responses indicate that you're doing well. Keep up with your healthy habits and continue taking good care of yourself. Remember, it's always a good idea to stay connected with loved ones and maintain a positive outlook. If you ever need support or someone to talk to, help is always just a message away.";
            }
            setMessages((prev) => [
              ...prev,
              {
                id: prev.length + 1,
                text: resultMessage,
                isBot: true,
              },
            ]);
            // Reset user responses list after the final answer is given
            setUserResponses([]);
            // Block any further inputs since the conversation is complete
            setConversationComplete(true);
          })
          .catch((error) => {
            setMessages((prev) => [
              ...prev,
              {
                id: prev.length + 1,
                text: "Error evaluating responses",
                isBot: true,
              },
            ]);
            setConversationComplete(true);
          });
      }

      const botMessage: Message = {
        id: messages.length + 2,
        text: nextMessage,
        isBot: true,
      };

      setMessages((prev) => [...prev, botMessage]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <main className="fixed inset-0 flex flex-col bg-gradient-to-b from-background to-secondary overflow-hidden">
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

      {/* Header */}
      <div className="relative z-10 w-full border-b bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <Button
            variant="ghost"
            className="flex items-center justify-center gap-2 text-muted-foreground transition-colors hover:text-foreground"
            onClick={() => router.push("/")}
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Button>
          <div className="text-sm text-muted-foreground flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Your conversation is private and secure
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <div className="relative z-10 flex-1 container mx-auto max-w-6xl px-4 py-4 flex flex-col h-[calc(100vh-8rem)]">
        <div className="flex-1 space-y-4 overflow-y-auto rounded-lg border bg-card/80 backdrop-blur-sm p-4 shadow-lg">
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className={`flex ${
                  message.isBot ? "justify-start" : "justify-end"
                }`}
              >
                <motion.div
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.2 }}
                  className={`flex max-w-[80%] items-start space-x-2 rounded-2xl px-4 py-3 ${
                    message.isBot
                      ? "bg-secondary text-secondary-foreground"
                      : "bg-primary text-primary-foreground"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-1 flex-shrink-0 rounded-full bg-background/10 p-1 flex items-center justify-center">
                      {message.isBot ? (
                        <Bot className="h-4 w-4" />
                      ) : (
                        <User className="h-4 w-4" />
                      )}
                    </div>
                    <p className="text-sm leading-relaxed">{message.text}</p>
                  </div>
                </motion.div>
              </motion.div>
            ))}
          </AnimatePresence>
          {/* Dummy element for auto-scroll */}
          <div ref={endOfMessagesRef} />

          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex justify-start"
            >
              <div className="flex max-w-[80%] items-start space-x-2 rounded-2xl bg-secondary px-4 py-3 text-secondary-foreground">
                <div className="flex items-center gap-2">
                  <div className="mt-1 flex-shrink-0 rounded-full bg-background/10 p-1 flex items-center justify-center">
                    <Bot className="h-4 w-4" />
                  </div>
                  <div className="flex space-x-1">
                    <motion.div
                      className="h-2 w-2 rounded-full bg-current"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                    <motion.div
                      className="h-2 w-2 rounded-full bg-current"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, delay: 0.2, repeat: Infinity }}
                    />
                    <motion.div
                      className="h-2 w-2 rounded-full bg-current"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, delay: 0.4, repeat: Infinity }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Input Area */}
        <div className="mt-4 flex items-center gap-3 bg-background/80 backdrop-blur-sm p-4 rounded-lg border">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
            placeholder="Type your message..."
            className="flex-1 rounded-full bg-card/80 backdrop-blur-sm px-4 py-2 shadow-sm focus-visible:ring-primary"
            disabled={conversationComplete}
          />
          <Button
            onClick={handleSend}
            size="icon"
            className="h-10 w-10 rounded-full flex items-center justify-center shrink-0"
            disabled={conversationComplete}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </main>
  );
}
